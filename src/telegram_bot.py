import asyncio
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
import os

from langgraph_agent import create_analytics_agent
from logger_config import (
    get_telegram_logger,
    log_success,
    log_error,
    log_info,
    log_warning,
)

# * Load environment variables
load_dotenv()

# * Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# * Validate required environment variables
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# * Initialize logger
logger = get_telegram_logger()

# * Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# * Initialize analytics agent
try:
    analytics_agent = create_analytics_agent()
    log_success(logger, "Analytics agent initialized successfully")
except Exception as e:
    log_error(logger, e, "Failed to initialize analytics agent")
    analytics_agent = None


def create_help_keyboard():
    """Create inline keyboard with help options."""
    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Sample Queries", callback_data="sample_queries")
    builder.button(text="📋 Available Metrics", callback_data="available_metrics")
    builder.adjust(1)
    return builder.as_markup()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    Handle /start command.
    """
    user_name = html.bold(
        message.from_user.full_name if message.from_user else "Unknown"
    )
    user_display = message.from_user.full_name if message.from_user else "Unknown User"
    log_info(logger, f"User {user_display} started the bot")

    welcome_text = f"""
🤖 <b>Welcome to Analytics Bot, {user_name}!</b>

I can help you analyze user data and calculate various metrics like active users by region.

<b>Available Commands:</b>
• /start - Show this welcome message
• /help - Get help and sample queries
• /status - Check bot status

<b>How to use:</b>
Just ask me questions about user analytics in natural language!

<i>Example: "How many active users were there in June 2024 by region?"</i>
    """

    await message.answer(welcome_text, reply_markup=create_help_keyboard())


@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    """
    Handle /help command.
    """
    user_display = message.from_user.full_name if message.from_user else "Unknown User"
    log_info(logger, f"User {user_display} requested help")
    
    help_text = """
📚 <b>How to Use Analytics Bot</b>

<b>Sample Questions You Can Ask:</b>
• "How many active users were there in June 2024 by region?
• "What was the conversion rate from registration to purchase for users who registered in June 2024?"
• "Calculate conversion rate for users registered between 2024-06-01 and 2024-06-15"
• "Show me the registration to purchase conversion metrics for June"
• "What's the average order check by region for June 2024?"
• "Calculate average order value by region from 2024-06-01 to 2024-06-30"
• "Show me regional spending patterns for the first half of June"
• "How many users registered but never made orders in June 2024?"
• "Calculate users without orders for the registration period 2024-06-01 to 2024-06-15"
• "What are the top 5 regions by registration count in June 2024?"
• "Show me the most popular regions for user registration"
• "Find the top 3 regions with highest registration volume"
• "What's the cancellation rate for June 2024?"
• "Calculate cancelled orders share from 2024-06-01 to 2024-06-30"
• "Show me the percentage of cancelled orders in June"
• "What's the customer lifetime value for June 2024 registrations?"
• "Calculate CLV for users registered from 2024-06-01 to 2024-06-30"
• "Show me the average lifetime value of customers"
• "What percentage of customers made repeat orders in June 2024?"
• "Calculate repeat customers percentage from 2024-06-01 to 2024-06-30"
• "Show me the percentage of users with multiple orders"
• "Show me registration dynamics for June 2024"
• "What are the registration trends from 2024-06-01 to 2024-06-30?"
• "Display daily registration patterns for June"
• "How many visitors didn't make purchases in June 2024?"
• "Count visitors without purchase from 2024-06-01 to 2024-06-30"
• "Show me the number of non-purchasing visitors"

<b>Available Metrics:</b>
• Active users by region for specific time periods
• Registration to purchase conversion rates
• Average order check (order value) by region
• Non-purchasing users (registered but never ordered) by region
• Top K regions by registration count
• Share of cancelled orders (cancellation rate)
• Customer Lifetime Value (CLV) for registered users
• Percentage of repeat customers (multiple orders)
• Registration dynamics over time (trends, peaks, growth patterns)
• Visitors without purchase (site visitors who didn't buy anything)

<b>Date Formats Supported:</b>
• YYYY-MM-DD (e.g., 2024-06-01)
• Natural language (e.g., "June 2024", "first half of June")

<b>Tips:</b>
• Be specific about date ranges for better results
• Use natural language - I understand context!
• Ask follow-up questions for more details
• Specify conversion windows for conversion rate analysis
• Specify number of top regions (default is 5)
    """
    
    await message.answer(help_text, reply_markup=create_help_keyboard())


@dp.message(Command("status"))
async def command_status_handler(message: Message) -> None:
    """
    Handle /status command.
    """
    user_display = message.from_user.full_name if message.from_user else "Unknown User"
    log_info(logger, f"User {user_display} requested status")

    if analytics_agent is None:
        status_text = "❌ <b>Bot Status: ERROR</b>\n\nAnalytics agent is not available."
    else:
        # * Check if data files exist
        data_file = Path("data/raw/users.csv")
        data_status = "✅ Available" if data_file.exists() else "❌ Missing"

        status_text = f"""
✅ <b>Bot Status: ONLINE</b>

<b>Components Status:</b>
• Analytics Agent: ✅ Active
• Data Files: {data_status}
• OpenAI API: ✅ Connected

<b>Data Info:</b>
• Users Data: {data_file}
• Last Updated: {data_file.stat().st_mtime if data_file.exists() else 'N/A'}

<b>Ready to answer your analytics questions!</b>
        """

    await message.answer(status_text)


@dp.callback_query(lambda c: c.data == "sample_queries")
async def process_sample_queries(callback_query):
    """Handle sample queries button."""
    user_display = callback_query.from_user.full_name if callback_query.from_user else "Unknown User"
    log_info(logger, f"User {user_display} requested sample queries")
    
    sample_text = """
📊 <b>Sample Analytics Queries</b>

<b>Active Users Analysis:</b>
1️⃣ "How many active users were there in June 2024 by region?"

2️⃣ "Calculate active users by region from 2024-06-01 to 2024-06-15"

<b>Conversion Rate Analysis:</b>
3️⃣ "What was the conversion rate from registration to purchase for users who registered in June 2024?"

4️⃣ "Calculate conversion rate for users registered between 2024-06-01 and 2024-06-15 with a 30-day window"

<b>Average Order Check Analysis:</b>
5️⃣ "What's the average order check by region for June 2024?"

6️⃣ "Show me regional spending patterns from 2024-06-01 to 2024-06-30"

<b>Non-Purchasing Users Analysis:</b>
7️⃣ "How many users registered but never made orders in June 2024?"

<b>Top Regions Analysis:</b>
8️⃣ "What are the top 5 regions by registration count in June 2024?"

9️⃣ "Show me the most popular regions for user registration from 2024-06-01 to 2024-06-15"

🔟 "Find the top 3 regions with highest registration volume"

<b>Cancelled Orders Analysis:</b>
1️⃣1️⃣ "What's the cancellation rate for June 2024?"

1️⃣2️⃣ "Calculate cancelled orders share from 2024-06-01 to 2024-06-30"

1️⃣3️⃣ "Show me the percentage of cancelled orders in June"

<b>Customer Lifetime Value Analysis:</b>
1️⃣4️⃣ "What's the customer lifetime value for June 2024 registrations?"

1️⃣5️⃣ "Calculate CLV for users registered from 2024-06-01 to 2024-06-30"

1️⃣6️⃣ "Show me the average lifetime value of customers"

<b>Repeat Customers Analysis:</b>
1️⃣7️⃣ "What percentage of customers made repeat orders in June 2024?"

1️⃣8️⃣ "Calculate repeat customers percentage from 2024-06-01 to 2024-06-30"

1️⃣9️⃣ "Show me the percentage of users with multiple orders"

<b>Registration Dynamics Analysis:</b>
2️⃣0️⃣ "Show me registration dynamics for June 2024"

2️⃣1️⃣ "What are the registration trends from 2024-06-01 to 2024-06-30?"

2️⃣2️⃣ "Display daily registration patterns for June"

<b>Visitors Without Purchase Analysis:</b>
2️⃣3️⃣ "How many visitors didn't make purchases in June 2024?"

2️⃣4️⃣ "Count visitors without purchase from 2024-06-01 to 2024-06-30"

2️⃣5️⃣ "Show me the number of non-purchasing visitors"

Just copy and paste any of these, or ask in your own words!
    """
    
    await callback_query.message.answer(sample_text)
    await callback_query.answer()


@dp.callback_query(lambda c: c.data == "available_metrics")
async def process_available_metrics(callback_query):
    """Handle available metrics button."""
    user_display = callback_query.from_user.full_name if callback_query.from_user else "Unknown User"
    log_info(logger, f"User {user_display} requested available metrics")
    
    metrics_text = """
📋 <b>Available Analytics Metrics</b>

<b>Currently Supported:</b>
• 📍 <b>Active Users by Region</b>
  - Count of active users grouped by geographical region
  - Supports custom date ranges
  - Shows regional distribution

• 📈 <b>Registration to Purchase Conversion Rate</b>
  - Percentage of registered users who made their first purchase
  - Configurable conversion window (default: 30 days)
  - Includes average days to purchase
  - Shows detailed conversion metrics

• 💰 <b>Average Order Check by Region</b>
  - Average order value grouped by geographical region
  - Shows spending patterns across regions
  - Includes order count and regional summaries
  - Supports custom date ranges for order analysis

• 🚫 <b>Non-Purchasing Users by Region</b>
  - Count of users who registered but never made orders
  - Grouped by geographical region
  - Shows conversion opportunities and lost potential
  - Includes non-purchasing rate percentage

• 🏆 <b>Top K Regions by Registration Count</b>
  - Ranked list of regions by registration volume
  - Configurable number of top regions (default: 5)
  - Shows registration counts and percentages
  - Identifies most popular regions for user acquisition

• ❌ <b>Share of Cancelled Orders</b>
  - Percentage of orders that were cancelled in a given period
  - Simple operational metric for service quality monitoring
  - Shows order fulfillment performance
  - Identifies periods with high cancellation rates

• 💰 <b>Customer Lifetime Value (CLV)</b>
  - Average total revenue generated per customer over their lifetime
  - Key metric for understanding customer value and profitability
  - Calculated for customers who registered in a specific period
  - Includes both purchasing and non-purchasing customers

• 🔄 <b>Repeat Customers Percentage</b>
  - Percentage of users who made more than 1 order in a given timeframe
  - Key metric for customer retention and loyalty analysis
  - Shows customer engagement and repeat purchase behavior
  - Identifies successful customer retention strategies

• 📈 <b>Registration Dynamics Over Time</b>
  - Time series analysis of user registration patterns
  - Shows daily, weekly, or monthly registration trends
  - Identifies peak registration periods and growth patterns
  - Provides insights into user acquisition effectiveness

• 👥 <b>Visitors Without Purchase</b>
  - Count of users who visited the site but didn't make purchases
  - Key metric for identifying conversion opportunities
  - Shows the gap between site traffic and actual sales
  - Helps measure conversion funnel effectiveness
    """
    
    await callback_query.message.answer(metrics_text)
    await callback_query.answer()


@dp.message()
async def handle_analytics_query(message: Message) -> None:
    """
    Handle analytics queries using the LangGraph agent.
    """
    if analytics_agent is None:
        user_display = (
            message.from_user.full_name if message.from_user else "Unknown User"
        )
        log_warning(logger, f"Service unavailable for user {user_display}")
        await message.answer(
            "❌ <b>Service Unavailable</b>\n\n"
            "The analytics service is currently unavailable. "
            "Please try again later or contact support."
        )
        return

    user_query = message.text
    user_name = message.from_user.full_name if message.from_user else "Unknown User"

    log_info(logger, f"Query from {user_name}: {user_query}")

    # * Send typing action
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # * Process query with analytics agent
        response = analytics_agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]}
        )

        # * Extract response content
        agent_response = response["messages"][-1].content

        # * Format response for Telegram
        formatted_response = f"""
🤖 <b>Analytics Result</b>

{agent_response}"""

        await message.answer(formatted_response)
        log_success(logger, f"Successfully processed query from {user_name}")

    except Exception as e:
        log_error(logger, e, f"Error processing query from {user_name}")

        error_response = f"""
❌ <b>Error Processing Query</b>

I encountered an issue while processing your request:
<code>{str(e)}</code>

<b>Suggestions:</b>
• Try rephrasing your question
• Check if you specified a valid date range
• Use /help for sample queries
• Contact support if the issue persists

<i>Example: "How many active users were there in June 2024 by region?"</i>
        """

        await message.answer(error_response)


async def on_startup():
    """Startup callback."""
    log_success(logger, "Telegram Analytics Bot started successfully!")
    log_info(
        logger,
        f"Analytics agent status: {'✅ Ready' if analytics_agent else '❌ Error'}",
    )


async def on_shutdown():
    """Shutdown callback."""
    log_info(logger, "Telegram Analytics Bot shutting down...")


async def main() -> None:
    """Main function to run the bot."""
    # * Register startup/shutdown callbacks
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # * Start polling
    log_info(logger, "Starting bot polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log_info(logger, "Bot stopped by user")
    except Exception as e:
        log_error(logger, e, "Fatal error occurred")
        sys.exit(1)
