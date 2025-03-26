# FinGenius API Documentation

The FinGenius Web API allows users to interact with the financial advisory system programmatically. Below are the available endpoints:

## Base URL
http://localhost:5000/api


## Endpoints

### 1. Analyze User Goal

**Endpoint**: `/analyze_goal`

**Method**: `POST`

**Description**: Processes a user's financial goal expressed in natural language.

**Request Body**:

```json
{
  "goal": "I want to save $50k for a down payment in 3 years with moderate risk."
}
```
Response:
```
{
  "time_horizon": 3,
  "target_amount": 50000,
  "risk_level": "moderate"
}

```
2. Fetch Market Data
Endpoint: /fetch_market_data
Method: GET

Description: Retrieves real-time market data for specified assets.
Query Parameters:
- symbols: Comma-separated list of asset symbols (e.g., AAPL,BTC,SPY)
- crypto: Boolean flag indicating if the assets include cryptocurrencies (e.g., true or false)

Response:
```
{
  "market_data": {
    "AAPL": {
      "price": 150.25,
      "volume": 1000000
    },
    "BTC": {
      "price": 45000.00,
      "volume": 5000
    },
    "SPY": {
      "price": 400.50,
      "volume": 2000000
    }
  }
}

```

3. Generate Investment Strategy
Endpoint: - /generate_strategy

Method: POST
Description: Creates a personalized investment strategy based on user goals and market data.

Request Body:
```
{
  "parsed_goal": {
    "time_horizon": 3,
    "target_amount": 50000,
    "risk_level": "moderate"
  },
  "market_data": {
    "AAPL": {
      "price": 150.25,
      "volume": 1000000
    },
    "BTC": {
      "price": 45000.00,
      "volume": 5000
    },
    "SPY": {
      "price": 400.50,
      "volume": 2000000
    }
  }
}

```
Response 
```
{
  "portfolio": {
    "AAPL": 0.4,
    "BTC": 0.3,
    "SPY": 0.3
  },
  "expected_return": 0.08,
  "risk": 0.15
}
```
4. Simulate Portfolio Performance
Endpoint: - /simulate_performance

Method: POST
Description: Simulates the performance of a given portfolio over a specified time horizon.
Request Body:

```
{
  "portfolio": {
    "AAPL": 0.4,
    "BTC": 0.3,
    "SPY": 0.3
  },
  "years": 3
}

```
Response 
```
{
  "final_value": 60000,
  "annualized_return": 0.08,
  "volatility": 0.15
}
```
## Error Handling
The API returns standard HTTP status codes to indicate the success or failure of a request. Error responses include a JSON object with an error field describing the issue.

Example:
```
{
  "error": "Invalid input data."
}
```
## Authentication
Currently, the API does not require authentication. Future versions may implement API key-based authentication.

## Rate Limiting
No rate limits are enforced at this time. However, this may change in future releases.

```

**3. `user_guide.md`**

```markdown
# FinGenius User Guide

Welcome to FinGenius, your AI-driven personalized financial advisor. This guide will help you set up and use FinGenius effectively.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Troubleshooting](#troubleshooting)
6. [FAQs](#faqs)
7. [Support](#support)

## 1. Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/adityakrmishra/FinGenius.git
   cd FinGenius

```
2. Install Dependencies:
   - pip install -r requirements.txt

2. Configuration
FinGenius requires API keys for data retrieval. Obtain keys from:

Alpha Vantage

CoinGecko

**Note**: It's a common practice to create a `.env.example` file that contains all the necessary environment variable keys without sensitive values. This serves as a template for others to understand which variables are required. :contentReference[oaicite:0]{index=0}

**Example `.env.example`**:

ALPHA_VANTAGE_API_KEY= COINGECKO_API_KEY= RISK_PROFILE=moderate



Users can copy this example file to create their own `.env` file and fill in the appropriate values. :contentReference[oaicite:1]{index=1}

## 3. Usage

FinGenius can be operated through both the Command-Line Interface (CLI) and the Web API.

### Command-Line Interface (CLI)

1. **Navigate to the Project Directory**:

   ```bash
   cd FinGenius

2. Run the Application:

- python src/main.py


3, Follow On-Screen Prompts: The CLI will guide you through inputting your financial goals and preferences.

## Web API
Start the Web Server:

Access API Documentation: Navigate to http://localhost:5000/docs in your web browser to explore available endpoints and their usage.

4. Examples
Example 1: Saving for a Down Payment
User Goal: "I want to save $50,000 for a down payment in 3 years with a moderate risk tolerance."

CLI Interaction:
```
$ python src/main.py
Welcome to FinGenius!
Please enter your financial goal:
> I want to save $50,000 for a down payment in 3 years with a moderate risk tolerance.
Analyzing your goal...
Time Horizon: 3 years
Target Amount: $50,000
Risk Level: Moderate
Generating investment strategy...
Recommended Portfolio:
- US Stocks: 50%
- Bonds: 30%
- International Stocks: 20%

```
### Web API Interaction:

Endpoint: POST /analyze_goal

Request Body:

```
{
  "goal": "I want to save $50,000 for a down payment in 3 years with a moderate risk tolerance."
}

```
Response:

```
{
  "time_horizon": 3,
  "target_amount": 50000,
  "risk_level": "moderate"
}

```
5. Troubleshooting
Issue: API keys not recognized.

Solution: Ensure that the .env file is correctly formatted and located in the project's root directory. Verify that the keys are accurate and have not expired.

Issue: ModuleNotFoundError for dotenv.

Solution: Install the python-dotenv library using pip install python-dotenv.

6. FAQs
Q: Can I use FinGenius for cryptocurrency investments?

A: Yes, FinGenius supports multiple asset classes, including cryptocurrencies. Ensure that you set the crypto flag to True when fetching market data.

Q: How does FinGenius assess my risk tolerance?

A: FinGenius uses a questionnaire to evaluate your risk profile, which influences the recommended investment strategy.

7. Support
- For additional support:

- Documentation: Refer to the docs/ directory for detailed guides and references.
  
- Issues: Report bugs or request features on the GitHub Issues page.

- Community: Join our discussion forum to connect with other users and developers.

Thank you for choosing FinGenius!
