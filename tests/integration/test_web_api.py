# tests/integration/test_web_api.py
import pytest
from fastapi.testclient import TestClient
from src.user_interface.web_api import app

@pytest.fixture
def test_client():
    return TestClient(app)

def test_api_healthcheck(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_portfolio_recommendation(test_client):
    test_payload = {
        "goal": "Retire with $1M in 10 years",
        "risk_profile": "moderate"
    }
    
    response = test_client.post("/recommend", json=test_payload)
    assert response.status_code == 200
    assert "portfolio" in response.json()
    assert sum(response.json()['portfolio'].values()) == pytest.approx(1.0)
