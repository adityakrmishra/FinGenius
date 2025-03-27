# tests/integration/test_rl_models.py
import pytest
import numpy as np
from src.reinforcement_learning.agents.dqn_agent import DQNAgent
from src.reinforcement_learning.environments.portfolio_env import PortfolioTradingEnv

@pytest.fixture
def sample_market_data():
    return np.random.rand(100, 5)  # 100 timesteps, 5 assets

def test_rl_environment(sample_market_data):
    env = PortfolioTradingEnv(
        market_data=sample_market_data,
        initial_balance=10000
    )
    
    state = env.reset()
    assert state.shape == (5,)  # 5 asset prices + portfolio state
    
    action = np.array([0.2, 0.3, 0.1, 0.2, 0.2])  # Portfolio allocation
    next_state, reward, done, _ = env.step(action)
    
    assert not done
    assert isinstance(reward, float)

def test_dqn_training_loop(sample_market_data):
    env = PortfolioTradingEnv(sample_market_data)
    agent = DQNAgent(state_size=env.observation_space.shape[0],
                    action_size=env.action_space.shape[0])
    
    initial_loss = agent.test(env)
    agent.train(env, episodes=10)
    final_loss = agent.test(env)
    
    assert final_loss < initial_loss
