# tests/unit/test_config.py
import pytest
from config import constants, paths

def test_config_constants():
    assert constants.HISTORICAL_WINDOW > 0
    assert len(constants.RISK_PROFILES) == 3
    assert '%' not in constants.FINANCIAL_TERMS

def test_path_resolution():
    raw_path = paths.RAW_DATA_DIR
    processed_path = paths.PROCESSED_DATA_DIR
    assert 'data/raw' in str(raw_path)
    assert raw_path.is_absolute() == False
    assert processed_path.exists()
