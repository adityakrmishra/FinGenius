import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """Dueling DQN architecture for portfolio management"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int = 128,
                 dueling: bool = True):
        super(DQNNetwork, self).__init__()
        
        self.dueling = dueling
        
        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Linear(hidden_dim, 1)
        
        if dueling:
            # Advantage stream
            self.advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        value = self.value(x)
        
        if self.dueling:
            advantage = self.advantage(x)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return value

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int = 128):
        super(PPONetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.policy(features), self.value(features)

class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid network for time series processing"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 output_dim: int):
        super(CNNLSTM, self).__init__()
        
        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv1d(input_shape[1], 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM Module
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # Conv1d expects (batch, features, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # LSTM expects (batch, seq_len, features)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
