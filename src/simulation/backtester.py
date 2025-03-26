"""
Backtesting Engine - Historical strategy performance analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from datetime import datetime

class Backtester:
    """Advanced backtesting engine for trading strategies"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtester with historical data and parameters
        
        Args:
            data: DataFrame containing prices and signals
            initial_capital: Starting portfolio value
            commission: Percentage commission per trade
            slippage: Percentage slippage per trade
            risk_free_rate: Annual risk-free rate for metrics
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.results = None

    def run_backtest(self) -> Dict:
        """Execute complete backtest"""
        # Initialize portfolio
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['capital'] = self.initial_capital
        portfolio['position'] = 0
        portfolio['total'] = self.initial_capital
        
        for i in range(1, len(self.data)):
            current_data = self.data.iloc[i]
            prev_data = self.data.iloc[i-1]
            
            # Execute trades
            position = current_data['signal']
            price = current_data['price']
            
            # Calculate slippage and commission impact
            trade_value = abs(position - portfolio['position'].iloc[i-1]) * price
            slippage_impact = trade_value * self.slippage
            commission_impact = trade_value * self.commission
            
            # Update portfolio
            portfolio.loc[portfolio.index[i], 'position'] = position
            portfolio.loc[portfolio.index[i], 'capital'] = (
                portfolio['capital'].iloc[i-1] 
                - slippage_impact 
                - commission_impact
            )
            portfolio.loc[portfolio.index[i], 'total'] = (
                portfolio['capital'].iloc[i] 
                + position * price
            )
        
        self.results = portfolio
        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        returns = self.results['total'].pct_change().dropna()
        cumulative_returns = (self.results['total'].iloc[-1] / self.initial_capital) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - self.risk_free_rate) / volatility
        max_drawdown = (self.results['total'] / self.results['total'].cummax() - 1).min()
        
        # Trade metrics
        num_trades = self.data['signal'].diff().abs().sum()
        
        return {
            'cumulative_return': cumulative_returns,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'final_value': self.results['total'].iloc[-1]
        }

# Example Usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=252)
    prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, 252))
    signals = np.random.choice([0, 1], size=252, p=[0.7, 0.3])
    
    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'signal': signals
    }).set_index('date')
    
    bt = Backtester(data)
    results = bt.run_backtest()
    print("Backtest Results:", results)
