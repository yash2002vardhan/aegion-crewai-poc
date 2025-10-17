# main.py
import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from utils.telegram import telegram_tool
from utils.slack import slack_tool
from utils.doc_search import code_docs_tool
from utils.kb_search import kb_search_tool
from crewai import Agent, Task, Crew
from utils.memory import QdrantMemoryWithMetadata
from datetime import datetime
import logging
from pydantic import BaseModel
import uvicorn

load_dotenv()  # loads .env

# config
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

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

# ==== MEMORY ARCHITECTURE ====
# We use a dual-memory system for optimal context retention:
#
# 1. CONVERSATION HISTORY (Short-term, in-memory)
#    - Stores last N messages per user for immediate context
#    - Fast access for recent conversation flow
#    - Explicitly passed in task description for agent awareness
#
# 2. QDRANT VECTOR DB (Long-term, persistent)
#    - Semantic search across all historical conversations
#    - Persists across restarts
#    - User-scoped retrieval for privacy
#
# 3. CREWAI BUILT-IN MEMORY (Agent-level)
#    - Enabled via memory=True in Agent config
#    - Handles internal task continuity
#
# All three work together: Conversation history provides immediate context,
# Qdrant provides relevant past context, and CrewAI memory handles reasoning continuity

qdrant_memory = QdrantMemoryWithMetadata(
    collection_name=os.getenv("QDRANT_COLLECTION", "customer_success_memory"),
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    embedding_model="text-embedding-3-large"
)

SHORT_TERM_WINDOW = 5
conversation_history = {}  # key: user_id -> list of recent messages

def add_to_conversation_history(user_id: str, message: str, role: str = "user"):
    """Add a message to the conversation history for context."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        "text": message, 
        "role": role,
        "timestamp": datetime.utcnow().isoformat()
    })
    # Keep only last N messages
    conversation_history[user_id] = conversation_history[user_id][-SHORT_TERM_WINDOW:]

def get_conversation_history(user_id: str) -> list:
    """Retrieve conversation history for a user."""
    return conversation_history.get(user_id, [])

# Agent setup
csm_agent = Agent(
    name="CustomerSuccessManager",
    role="Customer Success Manager",
    goal="Your job is NOT to write answers - it is to EXECUTE communication tools and return their confirmation messages.",
    backstory="""You are a customer success manager who MUST use communication tools to deliver responses.
    
    CRITICAL UNDERSTANDING:
    - Your deliverable is a tool confirmation message, NOT your own text
    - Generating an answer yourself is WRONG - you must CALL a tool
    - The tool will send the message and return "Successfully sent message..." - that's your output
    
    Your workflow:
    1. ANALYZE: Break message into safe vs sensitive topics
    2. SEARCH: Gather information from knowledge sources
    3. EXECUTE TOOL: Actually CALL "Send Response to Customer" or "Escalate to Human Expert"
    4. RETURN CONFIRMATION: The tool's "Successfully sent message..." IS your final answer
    
    You NEVER provide direct answers to customers. You ALWAYS use tools to send messages. 
    Your role is to decide WHAT to send and use the appropriate tool to send it.
    
    Example workflow:
    - Customer asks: "What is LLM?"
    - You search knowledge base
    - You formulate response: "LLM stands for Large Language Model..."
    - You CALL send_telegram_message tool with that text
    - Tool returns: "Successfully sent message to Telegram chat..."
    - That tool confirmation becomes your final answer
    
    For split responses, you may call BOTH tools (Telegram first, then Slack).""",
    tools=[kb_search_tool, code_docs_tool, telegram_tool, slack_tool],
    verbose=True,
    allow_delegation=False,
    memory=True,
    llm="gpt-4.1",
)

# guardrails: customize these to your needs
# NOTE: Currently not used - agent handles guardrails autonomously
# Keep this for potential future use if you want to add manual checks
def violates_guardrail(response: str) -> bool:
    lower = response.lower()
    forbidden_phrases = ["threat", "legal", "suicide", "self-harm", "refund denied"]
    return any(p in lower for p in forbidden_phrases)

def build_context_for_agent(user_id: str, incoming_message: str, top_k: int = 3):
    """
    Build context from recent conversation history and relevant past interactions.
    Returns formatted string for the agent's task description.
    """
    # Get recent conversation history
    history = get_conversation_history(user_id)
    if history:
        recent_text = "Recent conversation:\n"
        for i, msg in enumerate(history, 1):
            role_label = "Customer" if msg['role'] == 'user' else "Agent"
            recent_text += f"{i}. {role_label}: {msg['text'][:150]}\n"
    else:
        recent_text = "No recent conversation."

    # Get relevant past context from Qdrant (semantic search)
    past_hits = qdrant_memory.retrieve(incoming_message, top_k=top_k, user_id=user_id)
    if past_hits:
        past_text = "Relevant past interactions:\n"
        for i, hit in enumerate(past_hits, 1):
            payload = hit.get("payload", {})
            text = payload.get("text", "")[:150]
            role = payload.get("role", "unknown").upper()
            score = hit.get('score', 0)
            past_text += f"{i}. [{role}] {text}... (relevance: {score:.2f})\n"
    else:
        past_text = "No relevant past interactions."

    return f"""
--- RECENT CONVERSATION (Last {len(history)} messages) ---
{recent_text}

--- RELEVANT PAST CONTEXT ---
{past_text}
"""

# core process logic
def process_message_core(request: TicketRequest, is_bot_message: bool):
    """Process incoming customer messages and generate agent responses."""
    
    # If bot message, just store it for context and return early
    if is_bot_message:
        # Store bot's own messages for future context
        add_to_conversation_history(request.sender_id, request.message_content, role="assistant")
        qdrant_memory.add(
            text=request.message_content,
            metadata={
                "original_channel_id": request.original_channel_id,
                "sender_id": request.sender_id,
                "source": request.source,
                "message_id": request.message_id,
                "role": "assistant",
                "timestamp": request.timestamp
            }
        )
        return {"status": "success", "agent_result": "Bot message stored"}
    
    # Store user message in both conversation history and Qdrant
    add_to_conversation_history(request.sender_id, request.message_content, role="user")
    qdrant_memory.add(
        text=request.message_content,
        metadata={
            "original_channel_id": request.original_channel_id,
            "sender_id": request.sender_id,
            "source": request.source,
            "message_id": request.message_id,
            "role": "user",
            "timestamp": request.timestamp
        }
    )
    
    # Build context from conversation history and Qdrant
    context = build_context_for_agent(user_id=request.sender_id, incoming_message=request.message_content)

    # 4) Create a Task for the agent with autonomous tool usage
    task_description = f"""
    IMPORTANT: This task requires you to EXECUTE communication tools. For multi-topic messages with both 
    safe and sensitive topics, you may need to call BOTH tools to provide the best customer experience.
    
    CONVERSATION HISTORY AND CONTEXT:
    {context}
    
    CURRENT MESSAGE:
    Customer {request.sender_id} on {request.source} just sent: "{request.message_content}"
    
    STEP 1: ANALYZE AND CLASSIFY TOPICS
    Break down the message into individual topics/questions. For EACH topic, classify as:
    
    SAFE TOPICS (can answer directly):
    - Product features and functionality questions
    - Technical how-to questions
    - Configuration and setup questions
    - General support inquiries
    
    SENSITIVE TOPICS (must escalate):
    - Pricing/cost/payment/billing questions
    - Refund or cancellation requests
    - Threats, violence, self-harm, legal matters
    - Account deletion or termination
    - Security vulnerabilities or privacy concerns
    - Customer demanding human intervention
    
    STEP 2: SEPARATE INTO GROUPS
    - Group A (SAFE): Topics you can answer directly
    - Group B (SENSITIVE): Topics requiring escalation
    
    STEP 3: SEARCH FOR INFORMATION (for Group A topics)
    - Use "Search Product Documentation" for technical questions
    - Use "Search Knowledge Base" for general questions
    - Gather information to answer safe topics
    
    STEP 4: EXECUTE COMMUNICATION STRATEGY
    Choose based on what groups you have:
    
    SCENARIO 1 - ONLY SAFE topics (no sensitive):
    → CALL "Send Response to Customer" ONCE
    → Address all safe topics comprehensively
    → Wait for confirmation
    
    SCENARIO 2 - ONLY SENSITIVE topics (no safe):
    → CALL "Escalate to Human Expert" ONCE
    → Include all sensitive topics with reason, context, urgency
    → Wait for confirmation
    
    SCENARIO 3 - BOTH SAFE AND SENSITIVE topics:
    → CALL "Send Response to Customer" FIRST
      - Answer all safe topics
      - Mention that sensitive topics are being escalated
      - Wait for confirmation
    → THEN CALL "Escalate to Human Expert"
      - Include only the sensitive topics
      - Add reason, original message, context, urgency
      - Wait for confirmation
    
    CRITICAL RULES: 
    - You MUST actually EXECUTE tools (not describe them)
    - For Scenario 3, call BOTH tools in sequence (Telegram first, then Slack)
    - Each tool must return "Successfully sent message" confirmation
    - Task is complete only after all required tools have been executed
    
    ⚠️ IMPORTANT: Your "Final Answer" MUST be the tool confirmation message, NOT your own text.
    
    WRONG Final Answer: "LLM stands for Master of Laws..." (This is YOUR text - NOT ACCEPTABLE)
    CORRECT Final Answer: "Successfully sent message to Telegram chat -4886940973: 'LLM stands for...'" (This is the TOOL's confirmation)
    
    You must actually CALL the tool and receive its confirmation. The tool will return a message starting with 
    "Successfully sent message". That confirmation message IS your final answer. Do not generate your own final answer text.
    """
    
    support_task = Task(
        description=task_description,
        agent=csm_agent,
        expected_output="ONLY the tool confirmation message(s) that start with 'Successfully sent message to Telegram' or 'Successfully sent message to Slack'. Do NOT provide your own text - ONLY the tool's confirmation.",
        output_file=None,
        human_input=False
    )
    
    # 5) Create and execute the crew - Agent handles everything autonomously
    crew = Crew(
        agents=[csm_agent],
        tasks=[support_task],
        verbose=True,  # Enable to see agent reasoning and tool usage
        full_output=False  # Return only the final result, not intermediate steps
    )
    
    logger.info(f"Starting autonomous agent workflow for user {request.sender_id}")
    result = crew.kickoff()
    logger.info(f"Agent completed workflow. Result: {result}")
    
    # Store agent's response in memory for future context continuity
    agent_response = str(result)
    add_to_conversation_history(request.sender_id, agent_response, role="assistant")
    qdrant_memory.add(
        text=agent_response,
        metadata={
            "original_channel_id": request.original_channel_id,
            "sender_id": request.sender_id,
            "source": request.source,
            "message_id": f"{request.message_id}_response",
            "role": "assistant",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return {"status": "success", "agent_result": agent_response}


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
