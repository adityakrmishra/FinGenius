import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, Tuple, Optional

class PortfolioEnv(gym.Env):
    """Custom portfolio management environment for financial trading"""
    
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.0025,
                 reward_scaling: float = 1e-4):
        super(PortfolioEnv, self).__init__()
        
        self.data = data
        self.n_assets = data.shape[1] - 1  # Exclude timestamp
        self.current_step = 0
        
        # Action space: portfolio weights [0, 1] for each asset + cash
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Observation space: market data + portfolio state
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(low=-np.inf, high=np.inf, 
                                    shape=(self.n_assets, 5), dtype=np.float32),
            'portfolio': spaces.Box(low=0, high=np.inf, 
                                  shape=(self.n_assets + 1,), dtype=np.float32)
        })
        
        # Financial parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        
        self.reset()

    def reset(self) -> Dict:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.asset_prices = self._get_current_prices()
        self.portfolio = np.zeros(self.n_assets + 1)
        self.portfolio[-1] = self.balance  # Cash position
        
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one time step in the environment"""
        self.current_step += 1
        prev_portfolio_value = self._portfolio_value()
        
        # Normalize action to sum to 1
        action = action / (action.sum() + 1e-9)
        
        # Execute trades with transaction costs
        new_weights = action[:-1]
        current_weights = self.portfolio[:-1] / (self.portfolio.sum() + 1e-9)
        
        # Calculate transaction costs
        weight_diff = np.abs(new_weights - current_weights)
        transaction_cost = self.transaction_cost * weight_diff.sum()
        
        # Update portfolio
        self.portfolio = self._allocate_portfolio(action)
        self.portfolio[-1] -= transaction_cost  # Deduct transaction fees
        
        # Get new prices
        self.asset_prices = self._get_current_prices()
        
        # Calculate reward
        new_portfolio_value = self._portfolio_value()
        reward = (new_portfolio_value - prev_portfolio_value) * self.reward_scaling
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> Dict:
        """Construct observation dictionary"""
        return {
            'market_data': self._get_market_data(),
            'portfolio': self.portfolio.copy()
        }

    def _get_market_data(self) -> np.ndarray:
        """Get OHLCV data for current step"""
        return self.data.iloc[self.current_step, :-1].values.reshape(self.n_assets, 5)

    def _get_current_prices(self) -> np.ndarray:
        """Get current closing prices"""
        return self.data.iloc[self.current_step, ::5].values  # Close prices

    def _portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return (self.portfolio[:-1] * self.asset_prices).sum() + self.portfolio[-1]

    def _allocate_portfolio(self, action: np.ndarray) -> np.ndarray:
        """Convert action weights to actual portfolio allocation"""
        total_value = self._portfolio_value()
        allocation = total_value * action
        asset_quantities = allocation[:-1] / (self.asset_prices + 1e-9)
        cash = allocation[-1]
        return np.concatenate([asset_quantities, [cash]])

    def render(self, mode: str = 'human') -> None:
        """Render environment state"""
        value = self._portfolio_value()
        print(f"Step: {self.current_step} | Value: ${value:,.2f}")
