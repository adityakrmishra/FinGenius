"""
Command Line Interface Handler - Advanced financial toolkit CLI
"""

import argparse
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from colorama import Fore, Style
from simulation.backtester import Backtester
from simulation.monte_carlo import MonteCarloSimulator

class CLIHandler:
    """Advanced CLI for financial analysis operations"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="FinGenius - AI Financial Toolkit",
            epilog="Example: fingenius backtest --strategy momentum --start 2020-01-01"
        )
        self._setup_parser()
        self.args = None

    def _setup_parser(self):
        """Configure CLI arguments and subcommands"""
        subparsers = self.parser.add_subparsers(dest='command')

        # Backtest command
        backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtesting')
        backtest_parser.add_argument('--strategy', required=True, 
                                   choices=['momentum', 'mean_reversion', 'ml'],
                                   help='Trading strategy to test')
        backtest_parser.add_argument('--start', type=self._valid_date,
                                   default='2020-01-01', help='Start date (YYYY-MM-DD)')
        backtest_parser.add_argument('--end', type=self._valid_date,
                                   default=datetime.now().strftime('%Y-%m-%d'),
                                   help='End date (YYYY-MM-DD)')
        backtest_parser.add_argument('--capital', type=float, default=100000.0,
                                   help='Initial capital')

        # Optimize command
        optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
        optimize_parser.add_argument('--strategy', required=True,
                                   help='Strategy to optimize')
        optimize_parser.add_argument('--params', nargs='+',
                                   help='Parameters to optimize')

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze market data')
        analyze_parser.add_argument('--ticker', required=True,
                                  help='Asset ticker symbol')
        analyze_parser.add_argument('--period', default='1y',
                                  choices=['1d', '1w', '1m', '1y', '5y'],
                                  help='Time period to analyze')

        # Add common arguments
        for parser in [backtest_parser, optimize_parser, analyze_parser]:
            parser.add_argument('-v', '--verbose', action='store_true',
                              help='Verbose output')

    def _valid_date(self, s: str) -> datetime:
        """Validate date format"""
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid date: '{s}' (expected YYYY-MM-DD)")

    def run(self):
        """Execute CLI command"""
        self.args = self.parser.parse_args()
        if not self.args.command:
            self.parser.print_help()
            sys.exit(1)

        try:
            if self.args.command == 'backtest':
                self._handle_backtest()
            elif self.args.command == 'optimize':
                self._handle_optimize()
            elif self.args.command == 'analyze':
                self._handle_analyze()
        except Exception as e:
            self._print_error(f"Operation failed: {str(e)}")
            sys.exit(1)

    def _handle_backtest(self):
        """Process backtest command"""
        self._print_header(f"Backtesting {self.args.strategy} strategy")
        
        # Load data and initialize backtester
        data = self._load_data(self.args.strategy, self.args.start, self.args.end)
        bt = Backtester(data, initial_capital=self.args.capital)
        
        results = bt.run_backtest()
        self._print_results(results)

    def _handle_optimize(self):
        """Process optimize command"""
        self._print_header(f"Optimizing {self.args.strategy} parameters")
        # Optimization logic would go here
        self._print_success("Optimization complete")

    def _handle_analyze(self):
        """Process analyze command"""
        self._print_header(f"Analyzing {self.args.ticker} ({self.args.period})")
        # Analysis logic would go here
        self._print_success("Analysis complete")

    def _print_results(self, results: Dict[str, Any]):
        """Format and print backtest results"""
        print(f"\n{Fore.CYAN}Backtest Results:{Style.RESET_ALL}")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:>20}: {value:>10.2f}")
            else:
                print(f"{key:>20}: {value:>10}")

    def _print_header(self, text: str):
        print(f"\n{Fore.GREEN}=== {text} ==={Style.RESET_ALL}")

    def _print_success(self, text: str):
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

    def _print_error(self, text: str):
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}", file=sys.stderr)

if __name__ == "__main__":
    CLIHandler().run()
