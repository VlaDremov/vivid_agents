# Vivid Analytics Bot

An intelligent analytics bot supporting both Telegram and WhatsApp integrations for natural language analytics queries.

## 🚀 Features

- **Multi-platform support**: Telegram and WhatsApp bots
- **Natural language processing**: Ask analytics questions in plain English
- **Comprehensive analytics**: User metrics, conversion rates, CLV, and more
- **LangGraph integration**: Powered by advanced AI for query processing
- **Real-time data processing**: Instant responses to analytics queries
- **Robust logging**: Comprehensive logging and error handling

## 📊 Available Analytics

- Active users by region
- Registration to purchase conversion rates  
- Average order check by region
- Customer lifetime value (CLV)
- Repeat customer percentages
- Top regions by registration count
- Cancelled orders analysis
- Registration dynamics over time
- Non-purchasing user analysis

## 🛠 Setup

### Prerequisites

- Python 3.8+
- Telegram Bot Token (from @BotFather)
- Twilio Account (for WhatsApp)
- OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vivid_test
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Telegram Bot
   BOT_TOKEN=your_telegram_bot_token
   
   # OpenAI
   OPENAI_API_KEY=your_openai_api_key
   
   # Twilio WhatsApp (optional)
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   ```

4. **Generate sample data** (optional)
   ```bash
   python -m vivid_analytics.data.make_dummies
   ```

## 🚀 Running the Bots

### Telegram Bot
```bash
python -m vivid_analytics.bots.telegram_bot
```

### WhatsApp Bot (Twilio)
```bash
python -m vivid_analytics.bots.twilio_whatsapp_bot
```

## 💬 Sample Queries

- "How many active users were there in June 2024 by region?"
- "What was the conversion rate from registration to purchase for users who registered in June 2024?"
- "Calculate average order value by region for June 2024"
- "Show me the top 5 regions by registration count"
- "What's the customer lifetime value for June 2024 registrations?"
- "Calculate repeat customers percentage from 2024-06-01 to 2024-06-30"

## 📁 Project Structure

```
vivid_test/
├── README.md
├── requirements.txt
├── .env
├── vivid_analytics/           # Main package
│   ├── __init__.py
│   ├── analytics.py          # Core analytics functions
│   ├── langgraph_agent.py    # AI agent for query processing
│   ├── logger_config.py      # Logging configuration
│   ├── tools.py             # Utility tools
│   ├── bots/                # Bot implementations
│   │   ├── __init__.py
│   │   ├── telegram_bot.py   # Telegram bot
│   │   └── twilio_whatsapp_bot.py  # WhatsApp bot
│   └── data/                # Data generation and processing
├── tests/                   # Test files
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation
├── logs/                   # Application logs
└── data/                   # Data files
    └── raw/
        ├── users.csv
        └── orders.csv
```

## 🧪 Testing

Run tests using pytest:
```bash
python -m pytest tests/
```

## 📝 Logging

The application uses structured logging with different loggers for different components:
- Analytics operations
- Telegram bot activities  
- WhatsApp bot activities
- LangGraph agent operations

Logs are stored in the `logs/` directory with automatic rotation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`
- Open an issue on GitHub

## 🔧 Configuration

Additional configuration options can be found in the `config/` directory for advanced setups and customizations. 