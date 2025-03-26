import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Deque, Tuple, Optional

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
    
    def push(self, 
             state: Dict, 
             action: np.ndarray, 
             reward: float, 
             next_state: Dict, 
             done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        samples = random.sample(self.buffer, batch_size)
        return zip(*samples)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network agent for portfolio management"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 tau: float = 0.005,
                 buffer_size: int = 10000,
                 batch_size: int = 64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state: Dict, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.dirichlet(np.ones(self.action_dim))
        
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = torch.softmax(q_values, dim=-1).numpy()
        return action

    def update(self) -> Optional[float]:
        """Update policy network using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state_tensors = [self._state_to_tensor(s) for s in states]
        state_batch = torch.stack(state_tensors)
        # ... (rest of DQN update logic)
        
        # Update target network
        self._soft_update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def _soft_update_target_network(self) -> None:
        """Soft update target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def _state_to_tensor(self, state: Dict) -> torch.Tensor:
        """Convert state dictionary to network input tensor"""
        market_data = torch.FloatTensor(state['market_data'].flatten())
        portfolio = torch.FloatTensor(state['portfolio'])
        return torch.cat([market_data, portfolio])
