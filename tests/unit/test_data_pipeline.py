# tests/unit/test_data_pipeline.py
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.data_pipeline import data_ingestor, data_preprocessor

@pytest.fixture
def mock_api_client():
    mock = Mock()
    mock.fetch_data.return_value = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'volume': [1000, 2000, 3000, 4000, 5000]
    })
    return mock

def test_data_ingestor_fetch(mock_api_client):
    ingestor = data_ingestor.MarketDataIngestor(api_client=mock_api_client)
    df = ingestor.fetch_historical_data(symbol="AAPL", days=5)
    
    assert not df.empty
    assert len(df) == 5
    assert 'close' in df.columns
    mock_api_client.fetch_data.assert_called_once_with(symbol="AAPL", interval='1d', days=5)

def test_data_preprocessor_clean_data():
    raw_data = pd.DataFrame({
        'close': [100, 105, None, 110, 115],
        'volume': [1000, None, 3000, 3100, 3200]
    })
    
    processor = data_preprocessor.DataCleaner()
    cleaned = processor.clean_data(raw_data)
    
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == 5

def test_feature_engineering():
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 2000, 3000, 4000, 5000]
    })
    
    engineer = data_preprocessor.FeatureEngineer()
    features = engineer.add_technical_indicators(test_data)
    
    assert 'SMA_7' in features.columns
    assert 'RSI_14' in features.columns
    assert features['daily_return'].std() > 0
