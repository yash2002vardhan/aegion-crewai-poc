import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from crewai.tools import tool

load_dotenv(override=True)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")


@tool("send_slack_message")
def send_slack_message(text: str) -> str:
    """Escalate the query to human customer success team via Slack for review and response.
    
    ALWAYS use this tool when ANY of these conditions apply:
    1. BUSINESS queries (billing, pricing, subscriptions, contracts, account changes)
    2. SENSITIVE queries (compliance, legal, security, data privacy, policy questions)
    3. TECH SUPPORT queries (API errors, bugs, technical troubleshooting, system issues)
    4. LOW CONFIDENCE or UNCERTAIN (ambiguous KB, conflicting info, unclear user intent)
    5. GUARDRAIL TRIGGERED (empty KB results, can't find relevant info, safety concern)
    
    Format your message with: TYPE | CONTEXT | QUERY | SUGGESTED_RESPONSE. This ensures human review before customer sees the answer. When in doubt, ALWAYS escalate.
    
    Args:
        text: The message text to send to the Slack channel. The channel is automatically configured from environment.
        
    Returns:
        Success or error message indicating the result of the operation
    """
    # Use hardcoded channel from environment variable
    channel = SLACK_CHANNEL_ID
    
    if not channel:
        return "Error: SLACK_CHANNEL_ID environment variable is not set"
    
    token = SLACK_BOT_TOKEN
    if not token:
        return "Error: SLACK_BOT_TOKEN environment variable is not set"

    # Initialize the Slack client
    client = WebClient(token=token)

    try:
        # Call the chat.postMessage API
        response = client.chat_postMessage(
            channel=channel,
            text=text,
        )
        return f"Successfully sent message to Slack channel {channel}: '{text[:100]}...'"
    except SlackApiError as e:
        error_details = e.response.get('error', 'unknown_error')
        error_msg = f"Error sending message to Slack channel '{channel}': {error_details}"
        
        # Add helpful debugging info
        if error_details == "not_in_channel":
            error_msg += f"\nBot needs to be invited to channel {channel}. In Slack, type: /invite @your-bot-name"
        elif error_details == "channel_not_found":
            error_msg += f"\nChannel {channel} not found. Check SLACK_CHANNEL_ID in .env"
        elif error_details == "invalid_auth":
            error_msg += "\nInvalid token. Check SLACK_BOT_TOKEN in .env"
        
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error sending Slack message: {str(e)}"
        print(error_msg)
        return error_msg


# Expose the tool for use in CrewAI agents
slack_tool = send_slack_message

if __name__ == "__main__":
    # Test the Slack integration - channel is read from environment
    print(f"Testing Slack with channel: {SLACK_CHANNEL_ID}")
    result = send_slack_message(
        text="Hello, world! This is a test message from the customer success agent."
    )
    print(result)
