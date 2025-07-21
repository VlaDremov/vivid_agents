import os
from flask import Flask, request, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

from langgraph_agent import create_analytics_agent
from logger_config import (
    get_telegram_logger,  # * Reuse existing logger
    log_success,
    log_error,
    log_info,
)

# * Load environment variables
load_dotenv()

# * Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# * Validate required environment variables
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
    raise ValueError("Missing required Twilio environment variables")

# * Initialize logger (reuse existing)
logger = get_telegram_logger()

# * Initialize Flask app for webhooks
app = Flask(__name__)

# * Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# * Initialize analytics agent (unchanged!)
try:
    analytics_agent = create_analytics_agent()
    log_success(logger, "Analytics agent initialized successfully")
except Exception as e:
    log_error(logger, e, "Failed to initialize analytics agent")
    analytics_agent = None


def send_whatsapp_message(to: str, message: str) -> bool:
    """
    Send a WhatsApp message via Twilio.
    
    Args:
        to: Recipient WhatsApp number (format: whatsapp:+1234567890)
        message: Message text to send
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        # * Ensure the 'to' number has whatsapp: prefix
        if not to.startswith('whatsapp:'):
            to = f'whatsapp:{to}'
        
        twilio_message = twilio_client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to
        )
        
        log_success(logger, f"Message sent to {to}, SID: {twilio_message.sid}")
        return True
        
    except Exception as e:
        log_error(logger, Exception(f"Failed to send message to {to}: {str(e)}"), "Twilio API")
        return False


def format_phone_number(phone: str) -> str:
    """
    Format phone number for WhatsApp.
    
    Args:
        phone: Phone number (various formats)
        
    Returns:
        str: Formatted WhatsApp number (whatsapp:+1234567890)
    """
    # * Remove any existing whatsapp: prefix
    phone = phone.replace('whatsapp:', '')
    
    # * Add + if not present
    if not phone.startswith('+'):
        phone = f'+{phone}'
    
    return f'whatsapp:{phone}'


@app.route('/webhook', methods=['POST'])
def handle_whatsapp_webhook():
    """
    Handle incoming WhatsApp messages from Twilio.
    
    Twilio sends webhook data as form data, not JSON.
    """
    try:
        # * Get message data from Twilio webhook
        sender = request.form.get('From', '')  # * e.g., "whatsapp:+1234567890"
        message_body = request.form.get('Body', '')
        sender_name = request.form.get('ProfileName', 'Unknown User')
        
        log_info(logger, f"Message from {sender_name} ({sender}): {message_body}")
        
        # * Process with analytics agent (same logic as Telegram!)
        if analytics_agent and message_body:
            try:
                # * Same analytics processing as Telegram bot
                response = analytics_agent.invoke(
                    {"messages": [{"role": "user", "content": message_body}]}
                )
                
                agent_response = response["messages"][-1].content
                
                # * Format response for WhatsApp (simpler than Telegram HTML)
                formatted_response = f"""📊 *Analytics Result*

{agent_response}"""
                
                # * Send response via Twilio WhatsApp
                success = send_whatsapp_message(sender, formatted_response)
                
                if success:
                    log_success(logger, f"Successfully processed query from {sender_name}")
                else:
                    log_error(logger, Exception("Failed to send response"), "Message Sending")
                    
            except Exception as e:
                log_error(logger, e, f"Error processing query from {sender_name}")
                
                error_response = f"""❌ *Error Processing Query*

I encountered an issue while processing your request:
```{str(e)}```

*Suggestions:*
• Try rephrasing your question
• Check if you specified a valid date range
• Contact support if the issue persists

_Example: "How many active users were there in June 2024 by region?"_"""
                
                send_whatsapp_message(sender, error_response)
        else:
            # * Analytics agent not available or empty message
            if not analytics_agent:
                error_msg = "❌ *Service Unavailable*\n\nThe analytics service is currently unavailable. Please try again later."
                send_whatsapp_message(sender, error_msg)
            elif not message_body:
                welcome_msg = """👋 *Welcome to Analytics Bot!*

I can help you analyze user data and calculate various metrics.

*Sample Questions:*
• How many active users were there in June 2024 by region?
• What was the conversion rate from registration to purchase for users who registered in June 2024?
• Calculate average order value by region for June 2024
• Show me the top 5 regions by registration count

Just ask me your analytics questions in natural language!"""
                send_whatsapp_message(sender, welcome_msg)
        
        # * Return empty TwiML response (Twilio expects this)
        resp = MessagingResponse()
        return str(resp)
        
    except Exception as e:
        log_error(logger, e, "Error processing Twilio webhook")
        # * Return empty response even on error to avoid Twilio retries
        resp = MessagingResponse()
        return str(resp)


@app.route('/webhook', methods=['GET'])
def webhook_verification():
    """
    Webhook verification endpoint (if needed).
    Twilio typically doesn't require GET verification like some other services.
    """
    return "Webhook endpoint is active", 200


@app.route('/send_test', methods=['POST'])
def send_test_message():
    """
    Test endpoint to send messages manually.
    
    POST /send_test
    {
        "to": "+1234567890",
        "message": "Test message"
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        to = data.get('to')
        message = data.get('message')
        
        if not to or not message:
            return jsonify({"error": "Missing 'to' or 'message' parameter"}), 400
        
        # * Format phone number properly
        formatted_to = format_phone_number(to)
        success = send_whatsapp_message(formatted_to, message)
        
        if success:
            return jsonify({"status": "success", "to": formatted_to})
        else:
            return jsonify({"error": "Failed to send message"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "analytics_agent": "available" if analytics_agent else "unavailable",
        "twilio": "configured"
    })


def setup_twilio_webhook_url(webhook_url: str):
    """
    Configure Twilio to send webhooks to your URL.
    
    This can be done programmatically or via the Twilio Console.
    For sandbox, you typically configure this in the Console.
    """
    try:
        # * For production numbers, you can set webhook URL programmatically
        # * For sandbox, this is usually done in the Twilio Console
        log_info(logger, f"Webhook URL should be configured to: {webhook_url}")
        log_info(logger, "For Sandbox: Configure in Twilio Console > WhatsApp Sandbox")
        log_info(logger, "For Production: Can be set via API or Console")
        
    except Exception as e:
        log_error(logger, e, "Error setting up webhook URL")


if __name__ == "__main__":
    # * Optional: Set up webhook URL programmatically (mainly for production)
    webhook_url = os.getenv("WEBHOOK_URL")
    if webhook_url:
        setup_twilio_webhook_url(f"{webhook_url}/webhook")
    
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    log_success(logger, "Starting Twilio WhatsApp Analytics Bot...")
    log_info(logger, f"Twilio Account SID: {TWILIO_ACCOUNT_SID[:8] if TWILIO_ACCOUNT_SID else 'None'}...")
    log_info(logger, f"WhatsApp Number: {TWILIO_WHATSAPP_NUMBER}")
    log_info(logger, "Webhook endpoint: /webhook")
    log_info(logger, "Test endpoint: /send_test")
    log_info(logger, "Health endpoint: /health")
    
    app.run(host="0.0.0.0", port=port, debug=debug) 