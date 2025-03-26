import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
from typing import Dict, Tuple

class PPOAgent:
    """Proximal Policy Optimization agent for portfolio management"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 ppo_epochs: int = 4):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        
        # Policy and value networks
        self.policy = PPONetwork(state_dim, action_dim, hidden_dim)
        self.value_net = PPONetwork(state_dim, 1, hidden_dim)
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr)

    def act(self, state: Dict) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample action from current policy"""
        state_tensor = self._state_to_tensor(state)
        action_probs = self.policy(state_tensor)
        dist = Dirichlet(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy(), log_prob, dist.entropy().mean()

    def update(self, batch: Dict) -> Dict[str, float]:
        """Update policy using PPO clipped objective"""
        states = torch.stack([self._state_to_tensor(s) for s in batch['states']])
        # ... (rest of PPO update logic with GAE)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def _state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to network input tensor"""
        market_data = torch.FloatTensor(state['market_data'].flatten())
        portfolio = torch.FloatTensor(state['portfolio'])
        return torch.cat([market_data, portfolio])
