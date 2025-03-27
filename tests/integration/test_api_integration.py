# tests/integration/test_api_integration.py
import pytest
import os
from src.api_integration import alpha_vantage_client, coingecko_client

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('ALPHA_VANTAGE_API_KEY'),
    reason="API key required for integration tests"
)
def test_alpha_vantage_integration():
    client = alpha_vantage_client.AlphaVantageClient(os.getenv('ALPHA_VANTAGE_API_KEY'))
    data = client.get_daily_data(symbol="AAPL")
    
    assert isinstance(data, dict)
    assert 'Time Series (Daily)' in data
    assert len(data['Time Series (Daily)']) > 0

@pytest.mark.integration
def test_coingecko_api_connection():
    client = coingecko_client.CoinGeckoClient()
    status = client.check_api_status()
    
    assert status['gecko_says'] == '(V3) To the Moon!'
