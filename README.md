🧠 FutureTread – AI-Powered Trading Assistant
FutureTread is an AI-powered stock trading assistant that helps you:

📈 Predict buy/sell signals using machine learning

📊 Backtest strategies with historical data

🖥️ Interact via a powerful Streamlit dashboard

📲 Get Telegram alerts

🔁 Retrain the model from the UI

🧬 Use ensemble voting from multiple ML models

🚀 Features
Real-time price prediction using RandomForest

Built-in technical indicators: RSI, EMA, MACD

Streamlit UI with:

Signal visualization

Date filtering

Multi-ticker selection

Model retraining

Backtest stats

Telegram alerts for buy/sell

Model ensemble (RandomForest + Logistic + SVM)

Deployment-ready for Streamlit Cloud or HuggingFace

🛠️ Setup Instructions
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/AkashKumbhar07/FutureTread.git
cd FutureTread
2. Create a virtual environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On Mac/Linux
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Make sure you're using Python 3.11 for full compatibility.

▶️ Run the App
bash
Copy
Edit
streamlit run app.py
This will open the interactive dashboard in your browser.

🔁 Retrain the Model from UI
Press the “Retrain Model” button to:

Load latest data

Retrain using train_model.py

Save and reload models automatically

📲 Telegram Alerts Setup (Optional)
Create a Telegram bot using @BotFather

Get your bot token

Replace the token and your chat ID in utils/telegram_alert.py

python
Copy
Edit
BOT_TOKEN = "your_bot_token"
CHAT_ID = "your_chat_id"
🧬 Ensemble Voting
App uses multiple models (RandomForest, Logistic Regression, etc.)

Voting happens in utils/ensemble.py

Easily extendable by adding more models

📁 Project Structure
bash
Copy
Edit
FutureTread/
├── data/               # CSV files and saved price history
├── models/             # Trained models (*.pkl)
├── utils/
│   ├── indicators.py   # RSI, EMA, MACD
│   ├── train_model.py  # ML training logic
│   ├── ensemble.py     # Voting logic
│   └── telegram_alert.py
├── app.py              # Streamlit UI
├── backtest.py         # Backtesting script
├── main.py             # Data download script
├── live_predict.py     # Live signal prediction
└── requirements.txt
🌐 Deployment
You can deploy the app for free:

Streamlit Cloud

HuggingFace Spaces

🤝 Contributing
PRs and feedback welcome!
Let's build better AI traders together 🚀