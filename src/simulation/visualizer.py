"""
Data Visualization - Financial simulation results visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict

class Visualizer:
    """Advanced financial visualization toolkit"""
    
    def __init__(self, style: str = 'seaborn'):
        self.style = style
        plt.style.use(style)
        sns.set_palette("deep")

    def plot_equity_curve(self,
                        data: pd.DataFrame,
                        title: str = "Strategy Performance",
                        save_path: Optional[str] = None) -> None:
        """Plot portfolio value over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['total'], label='Portfolio Value')
        ax.plot(data['total'].cummax(), label='Peak Value', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        self._handle_output(fig, save_path)

    def plot_drawdown(self,
                    data: pd.DataFrame,
                    title: str = "Portfolio Drawdown",
                    save_path: Optional[str] = None) -> None:
        """Visualize maximum drawdown"""
        rolling_max = data['total'].cummax()
        drawdown = (data['total'] - rolling_max) / rolling_max
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(data.index, drawdown, 0, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        self._handle_output(fig, save_path)

    def plot_monte_carlo(self,
                        paths: np.ndarray,
                        title: str = "Monte Carlo Simulations",
                        save_path: Optional[str] = None) -> None:
        """Visualize Monte Carlo simulation paths"""
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(min(100, paths.shape[0])):
            ax.plot(paths[i], alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Portfolio Value")
        self._handle_output(fig, save_path)

    def _handle_output(self,
                      fig: plt.Figure,
                      save_path: Optional[str] = None) -> None:
        """Handle plot display/saving"""
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# Example Usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=252)
    portfolio_values = np.cumprod(1 + np.random.normal(0, 0.01, 252)) * 100000
    
    data = pd.DataFrame({
        'date': dates,
        'total': portfolio_values
    }).set_index('date')
    
    viz = Visualizer()
    viz.plot_equity_curve(data)
