import os
import httpx
from dotenv import load_dotenv
from crewai.tools import tool

load_dotenv(override=True)

TELEGRAM_ADAPTER_URL = os.getenv("TELEGRAM_ADAPTER_URL", "http://localhost:8001")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-4886940973")  # Default chat ID


@tool("Send Response to Customer")
def send_telegram_message(text: str) -> str:
    """Send a response message directly to the customer via Telegram.
    
    USE THIS TOOL FOR:
    - Product documentation questions
    - Technical how-to questions and tutorials
    - Feature explanations and capabilities
    - Follow-up questions and clarifications
    - General support inquiries
    - Configuration and setup questions
    - Troubleshooting technical issues
    - Questions about product functionality
    - Safe topics in multi-topic messages
    
    RESPONSE GUIDELINES:
    - Address all safe/answerable topics comprehensively
    - Be helpful and informative
    - Be transparent about limitations or knowledge gaps
    - Provide clear, actionable information
    - If message also contains sensitive topics that need escalation, mention that those are being handled separately
    
    SPLIT RESPONSE SCENARIO:
    - This tool can be used in combination with "Escalate to Human Expert"
    - First, answer safe topics with this tool
    - Then, escalate sensitive topics with the other tool
    - Both tools can be called for the same customer message when appropriate
    
    Args:
        text: The complete response message to send to the customer
        
    Returns:
        Confirmation message with send status (starts with "Successfully sent message")
    """
    # Use hardcoded chat_id from environment variable
    chat_id = TELEGRAM_CHAT_ID
    
    if not chat_id:
        return "Error: TELEGRAM_CHAT_ID environment variable is not set"

    if not text:
        return "Error: message text cannot be empty"

    # Use the escalate endpoint to send messages back to chats
    url = f"{TELEGRAM_ADAPTER_URL}/escalate"
    payload = {
        "text": text,
        "chat_id": chat_id
    }

    headers = {"Content-Type": "application/json"}
    timeout = httpx.Timeout(10.0, read=30.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                return f"Successfully sent message to Telegram chat {chat_id}: '{text}'"
            else:
                error_msg = f"Error sending message to Telegram: HTTP {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg
    except httpx.TimeoutException:
        error_msg = "Error: Request to telegram-adapter timed out"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error sending message to Telegram: {str(e)}"
        print(error_msg)
        return error_msg


# Expose the tool for use in CrewAI agents
telegram_tool = send_telegram_message


if __name__ == "__main__":
    # Example usage - chat_id is now read from environment variable
    result = send_telegram_message(
        text="Hello from agentic-lane! This is a test message."
    )
    print(result)
