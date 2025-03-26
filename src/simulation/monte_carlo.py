"""
Monte Carlo Simulation - Portfolio risk analysis through random sampling
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import norm, t

class MonteCarloSimulator:
    """Advanced Monte Carlo simulation engine"""
    
    def __init__(self,
                 returns: pd.Series,
                 num_simulations: int = 10000,
                 time_horizon: int = 252,
                 confidence_level: float = 0.95):
        """
        Initialize Monte Carlo simulator
        
        Args:
            returns: Historical return series
            num_simulations: Number of simulation paths
            time_horizon: Projection period in days
            confidence_level: VaR confidence level
        """
        self.returns = returns
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.confidence_level = confidence_level
        self.simulated_paths = None

    def run_simulation(self, method: str = 'normal') -> None:
        """Run Monte Carlo simulation"""
        if method == 'normal':
            self._normal_simulation()
        elif method == 'student-t':
            self._student_t_simulation()
        else:
            raise ValueError("Invalid simulation method")

    def _normal_simulation(self) -> None:
        """Normal distribution-based simulation"""
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        rand_returns = np.random.normal(
            mu, sigma,
            (self.num_simulations, self.time_horizon)
        )
        self.simulated_paths = np.cumprod(1 + rand_returns, axis=1)

    def _student_t_simulation(self) -> None:
        """Student's t-distribution-based simulation"""
        df, mu, sigma = t.fit(self.returns)
        
        rand_returns = t.rvs(
            df, loc=mu, scale=sigma,
            size=(self.num_simulations, self.time_horizon)
        )
        self.simulated_paths = np.cumprod(1 + rand_returns, axis=1)

    def calculate_var(self) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        final_returns = self.simulated_paths[:, -1] - 1
        var = np.percentile(final_returns, 100 * (1 - self.confidence_level))
        cvar = final_returns[final_returns <= var].mean()
        return var, cvar

# Example Usage
if __name__ == "__main__":
    # Generate sample returns
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    
    mc = MonteCarloSimulator(returns)
    mc.run_simulation(method='normal')
    var, cvar = mc.calculate_var()
    print(f"95% VaR: {var:.2%}, CVaR: {cvar:.2%}")
