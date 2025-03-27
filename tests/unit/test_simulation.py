# tests/unit/test_simulation.py
import pytest
import pandas as pd
from src.simulation import backtester, visualizer

@pytest.fixture
def sample_returns():
    return pd.Series([0.01, -0.02, 0.03, 0.015, -0.01], 
                   name='daily_returns')

def test_monte_carlo_simulation(sample_returns):
    simulations = backtester.run_monte_carlo(
        returns=sample_returns,
        years=3,
        num_sims=1000
    )
    
    assert simulations.shape == (756, 1000)  # 3 years * 252 days
    assert simulations.iloc[-1].mean() > 0

def test_visualization_output(tmpdir, sample_returns):
    output_path = tmpdir.mkdir("plots").join("returns.png")
    visualizer.plot_returns(sample_returns, str(output_path))
    
    assert output_path.exists()
    assert output_path.size() > 1024  # At least 1KB image
