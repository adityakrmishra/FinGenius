# config/constants.py

# API Configuration
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# NLP Model Constants
SPACY_MODEL_NAME = "en_core_web_sm"
FINANCIAL_TERMS = ["save", "invest", "retire", "growth", "income", "risk"]

# Risk Assessment
RISK_PROFILES = ["conservative", "moderate", "aggressive"]
RISK_QUESTIONS = [
    "What is your investment time horizon?",
    "How would you react to a 20% market drop?",
    "What is your target annual return?"
]

# Reinforcement Learning
RL_STATE_SIZE = 10  # Market features + portfolio state
RL_ACTION_SIZE = 5  # Asset allocation buckets
RL_EPISODES = 1000

# Data Processing
HISTORICAL_WINDOW = 252  # 1 year of trading days
