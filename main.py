# main.py
import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from utils.telegram import telegram_tool
from utils.slack import slack_tool
from utils.kb_search import kb_search_tool
from utils.milvus_search import mira_docs_tool
# from utils.github_search import github_search_tool  # Commented out - add GITHUB_TOKEN to .env to enable
from crewai import Agent, Task, Crew
from utils.memory import QdrantMemoryWithMetadata
from utils.todo_manager import TodoManager
from utils.todo_tools import (
    initialize_todo_tools,
    create_todo_list,
    update_todo_status,
    get_todo_list,
    get_next_pending_task
)
from datetime import datetime, timezone
import logging
from pydantic import BaseModel
import uvicorn

load_dotenv()  # loads .env

# config
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# initialize logging - suppress all logs except our custom output
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger("general_logger")

# Suppress verbose library logs (keep uvicorn for request logging)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("crewai").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

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


class PlanExecuteRequest(BaseModel):
    user_id: str
    message_content: str
    original_channel_id: str | None = None
    source: str | None = None

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

# ==== TODO MANAGER INITIALIZATION ====
# Initialize TodoManager for dynamic to-do list functionality
todo_manager = TodoManager()

# Initialize todo tools with the manager instance
initialize_todo_tools(todo_manager)

def add_to_conversation_history(user_id: str, message: str, role: str = "user"):
    """Add a message to the conversation history for context."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        "text": message, 
        "role": role,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    # Keep only last N messages
    conversation_history[user_id] = conversation_history[user_id][-SHORT_TERM_WINDOW:]

def get_conversation_history(user_id: str) -> list:
    """Retrieve conversation history for a user."""
    return conversation_history.get(user_id, [])

# ==== PLANNER & EXECUTOR AGENTS FOR TODO LIST ====
# These agents replace the old CSM agent with a clearer, more transparent workflow

# Planner Agent - Analyzes user requests and creates structured to-do lists
planner_agent = Agent(
    name="TaskPlanner",
    role="Task Planning Specialist with Guardrail Detection",
    goal="Analyze user requests, identify sensitive topics requiring escalation, and break them down into clear, sequential tasks with proper routing (Telegram for safe topics, Slack for sensitive topics).",
    backstory="""You are a strategic task planner who excels at breaking down complex requests and identifying which topics need human escalation.

    🚨 CRITICAL: GUARDRAIL & ESCALATION POLICY 🚨

    You MUST classify every user message into SAFE or SENSITIVE topics:

    ✅ SAFE TOPICS (Answer via Telegram):
    - Product features and functionality
    - Technical how-to questions
    - Configuration and setup instructions
    - General support inquiries
    - Documentation questions
    - Troubleshooting technical issues
    - API usage and examples
    - Mira product questions (use Search Mira Documentation tool)

    ⚠️ SENSITIVE TOPICS (Escalate to Slack - NEVER answer directly):
    - Pricing, billing, payments, invoices
    - Refunds or cancellations
    - Account deletion requests
    - Legal matters, lawsuits, threats
    - Self-harm or suicide mentions
    - Security vulnerabilities or data breaches
    - GDPR, privacy, or compliance requests
    - Explicit requests for human support
    - Abusive or harassing language

    Your task planning responsibilities:
    1. Analyze the user's request and identify ALL topics (both safe and sensitive)
    2. For SAFE topics: Create tasks to search for information and send response via Telegram
    3. For SENSITIVE topics: Create tasks to escalate to Slack (NEVER send sensitive info to Telegram)
    4. If message has BOTH: Create separate tasks for each (Telegram for safe, Slack for sensitive)
    5. Use the Create Todo List tool to generate the sequential to-do list

    Example 1 - SAFE topic only:
    User: "How do I reset my password?"
    Tasks:
    1. Search knowledge base for password reset instructions
    2. Send password reset instructions to customer via Telegram

    Example 1b - Mira question:
    User: "What is Mira?"
    Tasks:
    1. Search Mira documentation for product overview
    2. Send Mira information to customer via Telegram

    Example 2 - SENSITIVE topic only:
    User: "I want a refund immediately!"
    Tasks:
    1. Escalate refund request to Slack sales team with full context

    Example 3 - MIXED topics:
    User: "How do I enable 2FA? Also, what's your pricing?"
    Tasks:
    1. Search knowledge base for 2FA setup instructions
    2. Send 2FA instructions to customer via Telegram mentioning pricing team will follow up
    3. Escalate pricing question to Slack sales team with customer context

    REMEMBER: NEVER include pricing/billing/refund information in Telegram messages - ALWAYS escalate to Slack!

    After creating the task list, return it clearly so the Executor can work through it.
    """,
    tools=[create_todo_list, kb_search_tool, code_docs_tool, mira_docs_tool],
    verbose=False,  # Disable verbose logging - only show to-do list
    allow_delegation=False,
    memory=True,
    llm="gpt-4.1",
)

# Executor Agent - Executes tasks from the to-do list sequentially
executor_agent = Agent(
    name="TaskExecutor",
    role="Task Execution Specialist with Tool Routing",
    goal="Execute tasks from the to-do list one by one, using the correct tool (Telegram for customers, Slack for escalations), and ensuring all tasks are completed successfully.",
    backstory="""You are a methodical task executor who works through to-do lists systematically.

    Your workflow:
    1. Get the next pending task from the to-do list
    2. Mark the task as 'in_progress' before starting
    3. Execute the task using the appropriate tools
    4. Mark the task as 'completed' with the result
    5. Move to the next task and repeat
    6. Continue until all tasks are completed

    🔧 TOOL SELECTION RULES:

    Use TELEGRAM TOOL (Send Response to Customer) when task says:
    - "Send ... to customer via Telegram"
    - "Send response to customer"
    - "Send instructions to customer"
    - "Inform customer about..."

    Use SLACK TOOL (Escalate to Human Expert) when task says:
    - "Escalate ... to Slack"
    - "Send to Slack sales team"
    - "Escalate to human support"
    - "Forward to Slack channel"
    - Any mention of pricing/billing/refund escalation

    Use SEARCH TOOLS when task says:
    - "Search knowledge base..." → Use kb_search_tool
    - "Search documentation..." → Use code_docs_tool
    - "Search Mira documentation..." → Use mira_docs_tool (Search Mira Documentation)
    - "Find information about..." → Choose appropriate search tool based on context

    Execution principles:
    - Work on ONE task at a time (sequential execution)
    - ALWAYS update status before and after executing a task
    - Read the task description carefully to choose the right tool
    - Record what tools you used in the status update
    - If a task fails, mark it as 'failed' with an error message
    - Be thorough: complete each task fully before moving to the next

    Example execution flow for MIXED topics:
    1. Get Next Task → "Search knowledge base for 2FA setup"
    2. Mark 'in_progress' → Execute kb_search_tool
    3. Mark 'completed' with results
    4. Get Next Task → "Send 2FA instructions to customer via Telegram"
    5. Mark 'in_progress' → Execute telegram_tool (Send Response to Customer)
    6. Mark 'completed'
    7. Get Next Task → "Escalate pricing question to Slack"
    8. Mark 'in_progress' → Execute slack_tool (Escalate to Human Expert)
    9. Mark 'completed'

    Your final output should include a summary of all completed tasks.
    """,
    tools=[
        get_next_pending_task,
        update_todo_status,
        get_todo_list,
        kb_search_tool,
        code_docs_tool,
        mira_docs_tool,
        telegram_tool,
        slack_tool
    ],
    verbose=False,  # Disable verbose logging - only show to-do list
    allow_delegation=False,
    memory=True,
    llm="gpt-4.1",
)

# ==== ENDPOINTS ====

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/qdrant/get_all_data")
async def get_all_data():
    all_data = qdrant_memory.get_all_data()

    # Sort data by timestamp in ascending order
    sorted_data = sorted(all_data, key=lambda x: x.payload.get("timestamp", ""))

    return {"status": "success", "data": sorted_data, "count": len(sorted_data)}


def process_with_todo_list(user_id: str, message_content: str, original_channel_id: str = None, source: str = None):
    """
    Process message using Planner + Executor agents with to-do list.
    This is now the default workflow for ALL queries.
    """
    # Clear any existing to-do list for this user
    todo_manager.clear_list(user_id)

    # Task 1: Planning - Create the to-do list
    planning_task = Task(
        description=f"""
        Analyze the following user request and break it down into a sequential to-do list.

        User ID: {user_id}
        User Request: "{message_content}"
        Source: {source or "telegram"}
        Channel: {original_channel_id or "default"}

        Your job:
        1. Classify the message: Does it contain SAFE topics, SENSITIVE topics, or BOTH?
        2. For SAFE topics: Create tasks to search and respond via Telegram
        3. For SENSITIVE topics: Create tasks to escalate to Slack (pricing, billing, refunds, legal, etc.)
        4. Use the Create Todo List tool to create the to-do list
        5. Order tasks logically (search first, then respond/escalate)

        CRITICAL GUARDRAIL EXAMPLES:

        Example 1 - SAFE topic ("What is LLM?"):
        1. Search knowledge base for LLM information
        2. Search documentation for LLM details
        3. Send comprehensive response to customer via Telegram

        Example 1b - Mira question ("How does Mira work?"):
        1. Search Mira documentation for Mira functionality
        2. Send Mira explanation to customer via Telegram

        Example 2 - SENSITIVE topic ("How much does it cost?"):
        1. Escalate pricing inquiry to Slack sales team

        Example 3 - MIXED topics ("How do I reset password? Also need pricing info"):
        1. Search knowledge base for password reset instructions
        2. Send password reset instructions to customer via Telegram
        3. Escalate pricing inquiry to Slack sales team

        Create the to-do list now using the Create Todo List tool.
        """,
        agent=planner_agent,
        expected_output=f"A JSON response showing the created to-do list with task IDs, descriptions, and status for user {user_id}",
    )

    # Task 2: Execution - Work through the to-do list
    execution_task = Task(
        description=f"""
        Execute all tasks in the to-do list for user {user_id} sequentially.

        User Request Context: "{message_content}"
        Original Channel: {original_channel_id or "Not provided"}
        Source Platform: {source or "telegram"}

        Your workflow:
        1. Use Get Next Pending Task tool to get the first pending task
        2. Use Update Todo Status tool to mark it as 'in_progress'
        3. Execute the task:
           - If it's a search task, use kb_search_tool or code_docs_tool
           - If it's a messaging task, use telegram_tool or slack_tool
           - Use the appropriate tool based on the task description
        4. Use Update Todo Status tool to mark task as 'completed' with the result and tools_used
        5. Repeat steps 1-4 until no more pending tasks

        Important:
        - Complete each task fully before moving to the next
        - ALWAYS update status to 'in_progress' before executing
        - ALWAYS update status to 'completed' after executing
        - Record which tools you used and the result
        - If a task fails, mark it as 'failed' with the error message

        After completing all tasks, use Get Todo List tool to get the final state and provide a summary.
        """,
        agent=executor_agent,
        expected_output=f"A summary of all completed tasks with their results, and the final to-do list state showing all tasks as completed.",
    )

    # Create crew with sequential process
    crew = Crew(
        agents=[planner_agent, executor_agent],
        tasks=[planning_task, execution_task],
        verbose=False,  # Disable verbose - only show to-do list updates
        full_output=True,
    )

    # Execute the workflow
    result = crew.kickoff()

    # Get final to-do list state
    final_tasks = todo_manager.get_all_tasks(user_id)
    summary = todo_manager.get_summary(user_id)

    return {
        "status": "success",
        "user_id": user_id,
        "summary": summary,
        "tasks": final_tasks,
        "agent_output": str(result)
    }


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

    # If bot message, just store it
    if is_bot_message:
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

    try:
        # USE TODO-BASED WORKFLOW FOR ALL USER MESSAGES
        result = process_with_todo_list(
            user_id=request.sender_id,
            message_content=request.message_content,
            original_channel_id=request.original_channel_id,
            source=request.source
        )
        return result
    except Exception as e:
        logger.exception("Failed to process message")
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",   # Show uvicorn logs for requests
        access_log=True     # Enable access logging
    )
