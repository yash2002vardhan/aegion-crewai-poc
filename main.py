# main.py
import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from utils.telegram import telegram_tool
from utils.slack import slack_tool
from utils.doc_search import code_docs_tool
from utils.kb_search import kb_search_tool
# from utils.github_search import github_search_tool  # Commented out - add GITHUB_TOKEN to .env to enable
from crewai import Agent, Task, Crew
from utils.memory import QdrantMemoryWithMetadata
from datetime import datetime, timezone
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
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    # Keep only last N messages
    conversation_history[user_id] = conversation_history[user_id][-SHORT_TERM_WINDOW:]

def get_conversation_history(user_id: str) -> list:
    """Retrieve conversation history for a user."""
    return conversation_history.get(user_id, [])

# Agent setup
csm_agent = Agent(
    name="CustomerSuccessManager",
    role="Customer Success Manager with Chain-of-Thought Reasoning",
    goal="Execute communication tools with explicit reasoning and verification. NEVER generate direct responses - ALWAYS use tools and return their confirmation messages.",
    backstory="""You are a customer success manager who uses STRUCTURED CHAIN-OF-THOUGHT REASONING before every action.
    
    🧠 CHAIN-OF-THOUGHT PROTOCOL (ReAct Pattern):
    You MUST follow this reasoning cycle for EVERY customer message:
    
    THINK → ANALYZE → VERIFY → ACT → OBSERVE
    
    Before calling ANY tool, you must:
    1. STATE your understanding of the request
    2. LIST the information you need
    3. VERIFY you have all required parameters
    4. JUSTIFY your tool choice
    5. SCORE your confidence (0.0-1.0)
    6. ONLY THEN execute the tool
    
    CRITICAL UNDERSTANDING:
    - Your deliverable is a tool confirmation message, NOT your own text
    - Generating an answer yourself is WRONG - you must CALL a tool
    - The tool will send the message and return "Successfully sent message..." - that's your output
    - THINK BEFORE YOU ACT - explain your reasoning explicitly
    
    Your workflow with CoT:
    1. 🤔 THINK: "The customer is asking about X. This means I need to..."
    2. 📊 ANALYZE: Break message into safe vs sensitive topics with reasoning
    3. 🔍 SEARCH: Gather information and explain what you found
    4. ✅ VERIFY: Check you have all parameters (chat_id, message, etc.)
    5. 🎯 SCORE CONFIDENCE: Rate 0.0-1.0 how confident you are
    6. 🛠️ EXECUTE TOOL: Actually CALL the tool with verified parameters
    7. 👀 OBSERVE: The tool's "Successfully sent message..." IS your final answer
    
    CONFIDENCE SCORING RULES:
    - Score 0.8-1.0: High confidence, proceed with tool call
    - Score 0.5-0.8: Medium confidence, double-check parameters
    - Score <0.5: Low confidence, explain uncertainty and still proceed but note it
    
    Example workflow with CoT:
    Customer: "What is LLM?"
    
    [THOUGHT] The customer wants to know what LLM means. This is a technical/educational question.
    [ANALYSIS] Topic: LLM definition - SAFE (technical knowledge, no sensitive info)
    [SEARCH] Searching knowledge base for LLM information...
    [FOUND] LLM stands for Large Language Model, it's an AI system...
    [VERIFICATION] 
      ✓ I have the customer's chat_id
      ✓ I have drafted the message
      ✓ Source is Telegram
      ✓ Response is safe and accurate
    [CONFIDENCE] 0.95 - High confidence, clear technical question with verified answer
    [ACTION] Calling send_telegram_message tool...
    [OBSERVATION] Tool returned: "Successfully sent message to Telegram chat..."
    
    For split responses (safe + sensitive topics), use explicit reasoning:
    [THOUGHT] Message contains BOTH safe and sensitive topics
    [STRATEGY] Split response: Answer safe topics via Telegram, escalate sensitive via Slack
    [VERIFICATION] I will call TWO tools in sequence
    [ACTION 1] Calling send_telegram_message for safe topics...
    [OBSERVATION 1] Success confirmation received
    [ACTION 2] Calling send_slack_message for escalation...
    [OBSERVATION 2] Success confirmation received
    
    REMEMBER: Think step-by-step, verify parameters, score confidence, THEN act.""",
    tools=[kb_search_tool, code_docs_tool, telegram_tool, slack_tool],  # Removed github_search_tool - add GITHUB_TOKEN to enable
    verbose=True,
    allow_delegation=False,
    memory=True,
    llm="gpt-4.1",
)

# Quality Checker Agent - Verifies reasoning and tool usage
quality_checker_agent = Agent(
    name="QualityChecker",
    role="Quality Assurance Specialist",
    goal="Verify that the CSM agent used proper reasoning and actually called tools instead of generating direct responses.",
    backstory="""You are a quality checker who validates agent responses.
    
    Your job:
    1. Check if the agent followed chain-of-thought reasoning
    2. Verify that tools were actually CALLED (not just described)
    3. Confirm tool confirmation messages exist
    4. Flag any direct responses that bypass tools
    
    ACCEPTANCE CRITERIA:
    ✓ Response contains "Successfully sent message to Telegram" OR "Successfully sent message to Slack"
    ✓ Agent showed reasoning before acting
    ✓ No direct answers without tool calls
    
    REJECTION CRITERIA:
    ✗ Response is agent's own text without tool confirmation
    ✗ Agent described what to do but didn't do it
    ✗ Missing tool confirmation messages
    
    Output format:
    - APPROVED: [reason] if tools were properly used
    - REJECTED: [reason] if agent bypassed tools
    """,
    tools=[],
    verbose=True,
    allow_delegation=False,
    memory=False,
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

    # Get successful tool call patterns (helps with CoT reasoning)
    tool_pattern_hits = qdrant_memory.retrieve_by_metadata(
        query_text=incoming_message,
        filters={"role": "tool_success"},
        top_k=2
    )
    
    if tool_pattern_hits:
        tool_pattern_text = "\n💡 SIMILAR SUCCESSFUL TOOL CALLS FROM PAST:\n"
        for i, hit in enumerate(tool_pattern_hits, 1):
            payload = hit.get("payload", {})
            reasoning = payload.get("reasoning", "")[:200]
            tool_used = payload.get("tool_used", "unknown")
            tool_pattern_text += f"{i}. Past case used '{tool_used}': {reasoning}...\n"
    else:
        tool_pattern_text = ""

    return f"""
--- RECENT CONVERSATION (Last {len(history)} messages) ---
{recent_text}

--- RELEVANT PAST CONTEXT ---
{past_text}
{tool_pattern_text}
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

    # 4) Create a Task for the agent with Chain-of-Thought structure
    task_description = f"""
    🧠 CHAIN-OF-THOUGHT REASONING REQUIRED
    You MUST use structured reasoning before executing ANY tools. Follow the format below EXACTLY.
    
    ═══════════════════════════════════════════════════════════════
    📋 CONTEXT INFORMATION
    ═══════════════════════════════════════════════════════════════
    {context}
    
    📨 CURRENT MESSAGE:
    Customer ID: {request.sender_id}
    Platform: {request.source}
    Message: "{request.message_content}"
    
    ═══════════════════════════════════════════════════════════════
    🎯 YOUR STRUCTURED REASONING TASK
    ═══════════════════════════════════════════════════════════════
    
    You MUST complete ALL sections below before taking action:
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 1: INITIAL UNDERSTANDING                            │
    └─────────────────────────────────────────────────────────────┘
    [THOUGHT]
    State in your own words what the customer is asking for.
    Example: "The customer wants to know about pricing for the enterprise plan"
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 2: TOPIC CLASSIFICATION & ANALYSIS                  │
    └─────────────────────────────────────────────────────────────┘
    [ANALYSIS]
    Break down the message into individual topics. Classify each as SAFE or SENSITIVE:
    
    SAFE TOPICS (you can answer):
    ✓ Product features and functionality
    ✓ Technical how-to questions  
    ✓ Configuration and setup
    ✓ General support inquiries
    
    SENSITIVE TOPICS (must escalate):
    ⚠️ Pricing/billing/payments
    ⚠️ Refunds or cancellations
    ⚠️ Threats, legal, self-harm
    ⚠️ Account deletion
    ⚠️ Security/privacy concerns
    ⚠️ Explicit human intervention request
    
    Format your analysis:
    Topic 1: [description] → SAFE/SENSITIVE
    Topic 2: [description] → SAFE/SENSITIVE
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 3: INFORMATION GATHERING (if needed)                │
    └─────────────────────────────────────────────────────────────┘
    [SEARCH STRATEGY]
    For SAFE topics only, explain what information you need:
    - Will you search Product Documentation? Why?
    - Will you search Knowledge Base? Why?
    - What search terms will you use?
    
    [SEARCH RESULTS]
    After searching, summarize what you found.
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 4: TOOL SELECTION & JUSTIFICATION                   │
    └─────────────────────────────────────────────────────────────┘
    [TOOL DECISION MATRIX]
    
    Scenario Assessment:
    □ SCENARIO 1: Only SAFE topics → Use Telegram tool once
    □ SCENARIO 2: Only SENSITIVE topics → Use Slack tool once  
    □ SCENARIO 3: BOTH types → Use BOTH tools (Telegram first, then Slack)
    
    My scenario: [state which scenario applies]
    
    Tool(s) I will use:
    1. [Tool name] - Reason: [why this tool]
    2. [Tool name] - Reason: [why this tool] (if applicable)
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 5: PARAMETER VERIFICATION CHECKLIST                 │
    └─────────────────────────────────────────────────────────────┘
    [VERIFICATION]
    
    For Tool 1 ({request.source} communication):
    ✓ Customer chat_id available: {request.original_channel_id}
    ✓ Message drafted: [YES/NO - describe what you'll send]
    ✓ Message is appropriate: [YES/NO - explain why]
    ✓ All required parameters ready: [list them]
    
    For Tool 2 (Slack escalation) - if needed:
    ✓ Channel ID available: [state the channel]
    ✓ Escalation reason clear: [state reason]
    ✓ Context included: [YES/NO]
    ✓ Urgency level defined: [low/medium/high]
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 6: CONFIDENCE SCORING                               │
    └─────────────────────────────────────────────────────────────┘
    [CONFIDENCE SCORE]
    
    My confidence in this decision: [0.0 - 1.0]
    
    Justification:
    - I am confident because: [explain]
    - Potential risks: [list any concerns]
    - Mitigation: [how you've addressed concerns]
    
    Confidence interpretation:
    - 0.8-1.0: High confidence, proceed
    - 0.5-0.8: Medium confidence, double-checked
    - <0.5: Low confidence, noted uncertainty
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SECTION 7: EXECUTION PLAN                                   │
    └─────────────────────────────────────────────────────────────┘
    [ACTION PLAN]
    
    I will now execute the following actions IN ORDER:
    
    Action 1: CALL [tool name]
      - With message: [preview first 50 chars]
      - Expected outcome: Tool returns "Successfully sent message..."
    
    Action 2: CALL [tool name] (if applicable)
      - With message: [preview first 50 chars]  
      - Expected outcome: Tool returns "Successfully sent message..."
    
    ═══════════════════════════════════════════════════════════════
    📚 FEW-SHOT EXAMPLES (Learn from these)
    ═══════════════════════════════════════════════════════════════
    
    EXAMPLE 1 - Simple Technical Question (SCENARIO 1):
    ─────────────────────────────────────────────────────────────
    Customer: "How do I reset my password?"
    
    [THOUGHT] Customer needs password reset instructions.
    [ANALYSIS] Topic 1: Password reset → SAFE (technical support)
    [SEARCH STRATEGY] Search Knowledge Base for "password reset"
    [SEARCH RESULTS] Found: "Click 'Forgot Password' on login page..."
    [TOOL DECISION MATRIX] Scenario 1 - Only safe topics
    Tool: Send Response to Customer (Telegram)
    [VERIFICATION] ✓ chat_id: -123, ✓ Message ready, ✓ Appropriate
    [CONFIDENCE SCORE] 0.95 - Standard procedure, clear instructions
    [ACTION PLAN] Call send_telegram_message with instructions
    [EXECUTION] Calling tool now...
    [OBSERVATION] Successfully sent message to Telegram chat -123: "To reset..."
    
    EXAMPLE 2 - Sensitive Billing Question (SCENARIO 2):
    ─────────────────────────────────────────────────────────────
    Customer: "I want a refund immediately!"
    
    [THOUGHT] Customer is demanding a refund.
    [ANALYSIS] Topic 1: Refund request → SENSITIVE (financial matter)
    [SEARCH STRATEGY] No search needed - must escalate
    [TOOL DECISION MATRIX] Scenario 2 - Only sensitive topics
    Tool: Escalate to Human Expert (Slack)
    [VERIFICATION] ✓ Channel: sales-escalations, ✓ Reason: refund request
    [CONFIDENCE SCORE] 1.0 - Clear escalation policy
    [ACTION PLAN] Call send_slack_message with full context
    [EXECUTION] Calling tool now...
    [OBSERVATION] Successfully sent message to Slack channel sales-escalations...
    
    EXAMPLE 3 - Mixed Topics (SCENARIO 3):
    ─────────────────────────────────────────────────────────────
    Customer: "How do I enable 2FA? Also, what's your pricing?"
    
    [THOUGHT] Customer has two questions: technical + pricing.
    [ANALYSIS]
    Topic 1: 2FA setup → SAFE (technical feature)
    Topic 2: Pricing inquiry → SENSITIVE (financial)
    [SEARCH STRATEGY] Search docs for "2FA enable"
    [SEARCH RESULTS] Found: "Go to Settings > Security > Enable 2FA..."
    [TOOL DECISION MATRIX] Scenario 3 - BOTH types
    Tool 1: Send Response to Customer (Telegram) - for 2FA
    Tool 2: Escalate to Human Expert (Slack) - for pricing
    [VERIFICATION]
    Tool 1: ✓ chat_id, ✓ 2FA instructions ready, ✓ Will mention escalation
    Tool 2: ✓ channel, ✓ pricing question context, ✓ medium urgency
    [CONFIDENCE SCORE] 0.9 - Clear split strategy
    [ACTION PLAN]
    Action 1: Call send_telegram_message with 2FA answer + escalation note
    Action 2: Call send_slack_message with pricing question
    [EXECUTION]
    [ACTION 1] Calling send_telegram_message...
    [OBSERVATION 1] Successfully sent message to Telegram chat -456: "To enable 2FA... Regarding pricing, our team will contact you shortly."
    [ACTION 2] Calling send_slack_message...
    [OBSERVATION 2] Successfully sent message to Slack channel sales-escalations: "Customer asking about pricing..."
    
    ═══════════════════════════════════════════════════════════════
    ⚠️ CRITICAL EXECUTION RULES
    ═══════════════════════════════════════════════════════════════
    
    1. Complete ALL 7 sections above before calling tools
    2. Actually EXECUTE tools - don't just describe what you would do
    3. Your final answer MUST be the tool confirmation message(s)
    4. WRONG: "I will send a message about..." ← This is describing, not doing
    5. CORRECT: "Successfully sent message to Telegram..." ← This is the tool's response
    6. For Scenario 3, call BOTH tools and return BOTH confirmations
    7. Wait for each tool to return confirmation before moving to next tool
    
    ═══════════════════════════════════════════════════════════════
    🎬 NOW BEGIN YOUR STRUCTURED REASONING FOR THE CURRENT MESSAGE
    ═══════════════════════════════════════════════════════════════
    """
    
    support_task = Task(
        description=task_description,
        agent=csm_agent,
        expected_output="""Your final answer MUST contain at least one tool confirmation message.

        REQUIRED format - at least ONE of these MUST appear in your final answer:
        ✓ "Successfully sent message to Telegram chat [id]: [message preview]"
        ✓ "Successfully sent message to Slack channel [channel]: [message preview]"
        
        Your response MUST include:
        1. Your chain-of-thought reasoning (all 7 sections)
        2. The actual tool confirmation message(s) from executing the tool(s)
        
        WRONG final answer (missing tool confirmation):
        "I have analyzed the request and will send a message..." ❌
        
        CORRECT final answer (includes tool confirmation):
        "[reasoning process]... Successfully sent message to Telegram chat -123: 'Hello...' " ✓
        """,
        output_file=None,
        human_input=False
    )
    
    # 5) Create and execute the crew with output verification
    crew = Crew(
        agents=[csm_agent],
        tasks=[support_task],
        verbose=True,
        full_output=False
    )
    
    logger.info(f"Starting CoT agent workflow for user {request.sender_id}")
    
    # Verification and retry logic
    max_retries = 2
    attempt = 0
    verified_result = None
    
    while attempt < max_retries:
        attempt += 1
        logger.info(f"Attempt {attempt}/{max_retries}")
        
        result = crew.kickoff()
        result_str = str(result)
        
        # Verify that tools were actually called (check for confirmation messages)
        has_telegram_confirmation = "Successfully sent message to Telegram" in result_str
        has_slack_confirmation = "Successfully sent message to Slack" in result_str
        tool_was_called = has_telegram_confirmation or has_slack_confirmation
        
        if tool_was_called:
            logger.info(f"✓ Verification passed - Tool confirmation found")
            logger.info(f"  Telegram: {has_telegram_confirmation}, Slack: {has_slack_confirmation}")
            verified_result = result_str
            break
        else:
            logger.warning(f"✗ Verification failed - No tool confirmation found (attempt {attempt})")
            logger.warning(f"  Agent response: {result_str[:200]}...")
            
            if attempt < max_retries:
                logger.info(f"Retrying with enhanced prompt...")
                # Add stronger instruction for retry
                support_task.description += f"""
                
                ⚠️⚠️⚠️ RETRY INSTRUCTION ⚠️⚠️⚠️
                Your previous attempt did not include tool confirmation messages.
                You MUST actually CALL the tool (not just describe it) and include the tool's response.
                The tool will return a message like "Successfully sent message to Telegram chat..."
                That message MUST appear in your final answer.
                """
                crew = Crew(
                    agents=[csm_agent],
                    tasks=[support_task],
                    verbose=True,
                    full_output=False
                )
            else:
                logger.error(f"Max retries reached - Tool verification failed")
                verified_result = f"[VERIFICATION FAILED] Agent did not call tools properly. Raw response: {result_str}"
    
    logger.info(f"Agent completed workflow. Final result verified: {tool_was_called}")
    result = verified_result
    agent_response = str(result)
    
    # Store agent's response in memory for future context continuity
    add_to_conversation_history(request.sender_id, agent_response, role="assistant")
    qdrant_memory.add(
        text=agent_response,
        metadata={
            "original_channel_id": request.original_channel_id,
            "sender_id": request.sender_id,
            "source": request.source,
            "message_id": f"{request.message_id}_response",
            "role": "assistant",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
    
    # If tool was successfully called, store as a pattern for future CoT reasoning
    if tool_was_called:
        # Extract reasoning (everything before the tool confirmation)
        reasoning_parts = agent_response.split("Successfully sent message")
        reasoning = reasoning_parts[0] if len(reasoning_parts) > 1 else agent_response[:500]
        
        # Determine which tools were used
        tools_used = []
        if has_telegram_confirmation:
            tools_used.append("send_telegram_message")
        if has_slack_confirmation:
            tools_used.append("send_slack_message")
        
        # Store the successful pattern
        qdrant_memory.add(
            text=f"Customer query: {request.message_content}\nReasoning: {reasoning[:300]}",
            metadata={
                "original_channel_id": request.original_channel_id,
                "sender_id": request.sender_id,
                "source": request.source,
                "message_id": f"{request.message_id}_pattern",
                "role": "tool_success",
                "tool_used": ", ".join(tools_used),
                "reasoning": reasoning[:500],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        logger.info(f"Stored successful tool pattern: {tools_used}")
    
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
