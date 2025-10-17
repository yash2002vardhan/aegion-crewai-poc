import os
import httpx
from dotenv import load_dotenv
from crewai.tools import tool

load_dotenv(override=True)

TELEGRAM_ADAPTER_URL = os.getenv("TELEGRAM_ADAPTER_URL", "http://localhost:8001")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-4886940973")  # Default chat ID


@tool("send_telegram_message")
def send_telegram_message(text: str) -> str:
    """Send a warm, customer-facing response directly to the Telegram user.
    
    ONLY use this tool when ALL of these conditions are met:
    1. The query is GENERAL and NON-SENSITIVE (not business/billing/compliance/technical support)
    2. You have HIGH CONFIDENCE in your answer from the knowledge base
    3. The KB results are CLEAR, NON-CONFLICTING, and SUFFICIENT to answer
    4. NO guardrails were triggered (no empty/ambiguous/conflicting KB results)
    
    This sends the response directly to the customer. If you have ANY doubt, use send_slack_message instead to escalate for human review.
    
    Args:
        text: The message text to send to the Telegram user. The chat_id is automatically configured from environment.
        
    Returns:
        Success or error message indicating the result of the operation
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
