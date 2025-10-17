# main.py
import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from utils.telegram import telegram_tool
from utils.slack import slack_tool
from crewai import Agent, Task, Crew
from crewai_tools import QdrantVectorSearchTool
from utils.memory import QdrantMemoryWithMetadata
from datetime import datetime
import logging
from pydantic import BaseModel
import uvicorn

load_dotenv()  # loads .env

# config
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))

# initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csm_agent")

app = FastAPI(title="CrewAI CSM Agent")


class TicketRequest(BaseModel):
    original_channel_id: str
    sender_id: str
    source: str
    message_id: str
    message_content: str
    timestamp: str
    channel_id: str | None = None
    ai_response: str | None = None

# ==== MEMORY ====
qdrant_memory = QdrantMemoryWithMetadata(
    collection_name=os.getenv("QDRANT_COLLECTION", "customer_success_memory"),
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    embedding_model="text-embedding-3-large"
)

# short-term memory: simple list (last N messages). We'll keep a per-conversation rolling window in memory.
SHORT_TERM_WINDOW = 5
short_term_cache = {}  # key: conversation_id/user_id -> list of messages

def add_to_short_term(user_key: str, message: str):
    lst = short_term_cache.get(user_key, [])
    lst.append({"text": message, "timestamp": datetime.utcnow().isoformat()})
    # keep last N
    short_term_cache[user_key] = lst[-SHORT_TERM_WINDOW:]

def get_short_term_context(user_key: str):
    return short_term_cache.get(user_key, [])

# optional KB tool
kb_tool = QdrantVectorSearchTool(
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY", None),
    collection_name=os.getenv("KB_COLLECTION", "customer_success_memory")
)

# Agent setup
csm_agent = Agent(
    name="CustomerSuccessManager",
    role="Customer Success Manager",
    goal="Resolve customer inquiries using available knowledge and communication tools.",
    backstory="""You are an experienced customer success manager with access to a knowledge base and communication tools. 
    You understand when to handle queries directly and when to escalate to human experts.""",
    tools=[kb_tool, telegram_tool, slack_tool],
    verbose=True,
    allow_delegation=False,
    # memory=True,  # Enable CrewAI's built-in short-term memory (optional - currently using custom)
    llm="gpt-4.1-mini",
)

# guardrails: customize these to your needs
# NOTE: Currently not used - agent handles guardrails autonomously
# Keep this for potential future use if you want to add manual checks
def violates_guardrail(response: str) -> bool:
    lower = response.lower()
    forbidden_phrases = ["threat", "legal", "suicide", "self-harm", "refund denied"]
    return any(p in lower for p in forbidden_phrases)

# helper to call agent: we provide context manually using memory + short-term
def build_context_for_agent(user_id: str, incoming_message: str, top_k: int = 5):
    """
    Returns a string context combining short-term and long-term memory entries.
    """
    short_term = get_short_term_context(user_id)
    short_text = "\n".join([f"{m['timestamp']}: {m['text']}" for m in short_term]) if short_term else ""

    # long-term (user-scoped retrieval)
    long_term_hits = qdrant_memory.retrieve(incoming_message, top_k=top_k, user_id=user_id)
    long_texts = []
    for hit in long_term_hits:
        payload = hit.get("payload", {})
        ts = payload.get("timestamp", "")
        text = payload.get("text", "")
        long_texts.append(f"{ts}: {text} (score={hit.get('score')})")
    long_text = "\n".join(long_texts)

    # Compose a single context blob (you can refine prompt engineering later)
    combined = "\n\n--- SHORT-TERM CONTEXT ---\n" + (short_text or "None") + \
               "\n\n--- LONG-TERM CONTEXT ---\n" + (long_text or "None") + \
               f"\n\n--- INCOMING MESSAGE ---\n{incoming_message}\n"

    return combined

# core process logic
def process_message_core(request: TicketRequest, is_bot_message:bool):

    if is_bot_message:
        metadata = {
            "original_channel_id": request.original_channel_id,
            "sender_id": request.sender_id,
            "source": request.source,
            "message_id": request.message_id,
            "role": "assistant",
            "timestamp": request.timestamp
        }

        qdrant_memory.add(text=request.message_content, metadata=metadata)
        add_to_short_term(request.sender_id, request.message_content)

        return {"status": "success", "agent_result": "Bot message ignored"}

    
    metadata = {
            "original_channel_id": request.original_channel_id,
            "sender_id": request.sender_id,
            "source": request.source,
            "message_id": request.message_id,
            "role": "user",
            "timestamp": request.timestamp
    }
    qdrant_memory.add(text=request.message_content, metadata=metadata)

    # 2) add to short-term cache
    add_to_short_term(request.sender_id, request.message_content)

    # 3) build context
    context = build_context_for_agent(user_id=request.sender_id, incoming_message=request.message_content, top_k=5)

    # 4) Create a Task for the agent with autonomous tool usage
    task_description = f"""
    Customer {request.sender_id} on {request.source} sent: "{request.message_content}"
    
    Resolve this inquiry by responding appropriately. Use your tools as needed.
    """
    
    support_task = Task(
        description=task_description,
        agent=csm_agent,
        expected_output="Confirmation that the inquiry was handled (either response sent or escalation completed)"
    )
    
    # 5) Create and execute the crew - Agent handles everything autonomously
    crew = Crew(
        agents=[csm_agent],
        tasks=[support_task],
        verbose=True  # Enable to see agent reasoning and tool usage
    )
    
    logger.info(f"Starting autonomous agent workflow for user {request.sender_id}")
    result = crew.kickoff()
    
    logger.info(f"Agent completed workflow. Result: {result}")

    
    return {"status": "success", "agent_result": str(result)}


@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/qdrant/get_all_data")
async def get_all_data():
    all_data = qdrant_memory.get_all_data()
    
    # Sort data by timestamp in ascending order
    sorted_data = sorted(all_data, key=lambda x: x.payload.get("timestamp", ""))
    
    return {"status": "success", "data": sorted_data, "count": len(sorted_data)}


@app.post("/process")
async def process(request: TicketRequest):
    
    if not request.sender_id or not request.message_content:
        return {"error": "Missing sender_id or message_content in request body"}, 400

    BOT_ID = os.getenv("BOT_ID", "bot")
    
    # Check if this is a bot message (multiple indicators)
    is_bot_message = (
        request.sender_id == BOT_ID or  # Direct bot ID check
        len(request.message_content) > 200 or  # Bot responses are usually long
        request.message_content.startswith("Hello! You") or  # Common bot greeting pattern
        "If you have more questions" in request.message_content  # Bot signature phrase
    )
    
    try:
        result = process_message_core(request=request, is_bot_message=is_bot_message)
        return result
    except Exception as e:
        logger.exception("Failed to process message")
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
