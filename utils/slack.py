import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from crewai.tools import tool

load_dotenv(override=True)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")


@tool("Escalate to Human Expert")
def send_slack_message(text: str) -> str:
    """Escalate customer inquiries to human experts via Slack for situations requiring human judgment.
    
    USE THIS TOOL WHEN THE MESSAGE CONTAINS:
    
    PRICING AND FINANCIAL QUESTIONS:
    - Questions about price, pricing, or cost
    - Payment or billing inquiries
    - Subscription or plan questions
    - Refund requests
    - Cancellation requests
    - Invoice or charge inquiries
    
    GUARDRAIL SITUATIONS:
    - Threats or threatening language
    - Legal matters, lawsuits, or lawyer mentions
    - Self-harm or suicide references
    - Abusive or harassing language
    - Mentions of illegal activities
    
    SENSITIVE BUSINESS MATTERS:
    - Account suspension or termination requests
    - Account deletion requests
    - Data privacy or GDPR requests
    - Compliance inquiries
    - Security vulnerability reports
    - Data breach concerns
    - Contract or SLA discussions
    
    CRITICAL TECHNICAL ISSUES:
    - System-wide outages
    - Critical bugs causing data loss
    - Production environment failures
    - Business-critical disruptions
    
    CUSTOMER ESCALATION REQUESTS:
    - Customer explicitly requests to speak to a human
    - Customer asks for manager or supervisor
    - Customer explicitly requests escalation
    - Extremely frustrated tone with serious complaints
    
    SPLIT RESPONSE SCENARIO:
    - This tool can be used in combination with "Send Response to Customer"
    - If message has both safe and sensitive topics, first answer safe topics via Telegram
    - Then use this tool to escalate only the sensitive topics
    - Both tools can be called for the same customer message when appropriate
    
    ESCALATION MESSAGE FORMAT - Include in your text:
    1. ESCALATION REASON: Which category triggered the escalation
    2. SENSITIVE TOPICS: The specific sensitive parts requiring escalation
    3. CUSTOMER'S ORIGINAL MESSAGE: Full text of the message
    4. CONTEXT: Any relevant conversation history
    5. URGENCY LEVEL: Indicate high, medium, or low urgency
    
    Args:
        text: Complete escalation summary with reason, sensitive topics, original message, context, and urgency level
        
    Returns:
        Confirmation message with send status (starts with "Successfully sent message to Slack")
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
