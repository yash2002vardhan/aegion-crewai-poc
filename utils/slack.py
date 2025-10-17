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
    """Escalate a customer inquiry to the human support team via Slack.
    
    Use this when:
    - Query involves billing, pricing, payments, or account changes
    - Query is about technical issues, bugs, API errors, or system problems
    - Query involves compliance, legal, security, or data privacy matters
    - Knowledge base has no relevant information or results are conflicting
    - You're uncertain about the correct answer
    - The topic requires expert human judgment
    
    Your message should include:
    - Query type/category
    - Customer's question
    - Relevant context from knowledge base (if any)
    - Why escalation is needed
    
    Args:
        text: Summary and context for the human team to review
        
    Returns:
        Confirmation message with send status
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
