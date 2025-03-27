# tests/unit/test_rl_trainer.py
import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.reinforcement_learning import trainer

@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.train.return_value = {'episode_reward': 100}
    agent.test.return_value = {'mean_reward': 120}
    return agent

def test_training_progress(mock_agent):
    env = Mock()
    env.reset.return_value = np.random.rand(10)
    env.step.return_value = (np.random.rand(10), 1.5, False, {}
    
    training_cfg = {
        'episodes': 100,
        'eval_freq': 20,
        'checkpoint_dir': 'models/checkpoints'
    }
    
    results = trainer.run_training(mock_agent, env, training_cfg)
    
    assert results['final_reward'] > 0
    mock_agent.save.assert_called_once_with('models/checkpoints/final_model.pt')

def test_hyperparameter_validation():
    with pytest.raises(ValueError):
        trainer.validate_config({
            'learning_rate': -0.01,
            'buffer_size': 'invalid'
        })
