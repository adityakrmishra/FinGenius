"""
FinGenius Main Module - Core application entry point
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import yaml

from data_pipeline.data_ingestor import DataIngestor
from data_pipeline.data_preprocessor import DataPreprocessor
from reinforcement_learning.trainer import Trainer
from reinforcement_learning.environments.portfolio_env import PortfolioEnv
from user_interface.cli_handler import CLIHandler
from user_interface.web_api import app as web_app
from simulation.backtester import Backtester
from simulation.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fingenius.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path('config.yaml')

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load application configuration from YAML file"""
    try:
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def setup_arg_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='FinGenius - AI-Driven Financial Analysis Platform',
        epilog='Example: python main.py data --source alphavantage --symbols MSFT,AAPL'
    )

    # Global arguments
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH,
                      help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable debug logging')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Data pipeline command
    data_parser = subparsers.add_parser('data', help='Data management operations')
    data_subparsers = data_parser.add_subparsers(dest='data_command')
    
    # Data ingestion
    ingest_parser = data_subparsers.add_parser('ingest', help='Ingest financial data')
    ingest_parser.add_argument('--source', required=True,
                             choices=['alphavantage', 'coingecko', 'csv'],
                             help='Data source')
    ingest_parser.add_argument('--symbols', required=True,
                             help='Comma-separated list of symbols')
    ingest_parser.add_argument('--output', type=Path, default='data/raw',
                             help='Output directory')
    
    # Data preprocessing
    preprocess_parser = data_subparsers.add_parser('preprocess', 
                                                 help='Preprocess raw data')
    preprocess_parser.add_argument('--input', type=Path, required=True,
                                 help='Input data directory')
    preprocess_parser.add_argument('--output', type=Path, default='data/processed',
                                 help='Output directory')

    # RL training command
    rl_parser = subparsers.add_parser('rl', help='Reinforcement learning operations')
    rl_subparsers = rl_parser.add_subparsers(dest='rl_command')
    
    # RL training
    train_parser = rl_subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--agent', required=True,
                            choices=['dqn', 'ppo'],
                            help='RL agent type')
    train_parser.add_argument('--episodes', type=int, default=1000,
                            help='Number of training episodes')
    train_parser.add_argument('--data', type=Path, required=True,
                            help='Path to training data')
    
    # RL evaluation
    eval_parser = rl_subparsers.add_parser('evaluate', help='Evaluate RL agent')
    eval_parser.add_argument('--model', type=Path, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--episodes', type=int, default=100,
                           help='Number of evaluation episodes')

    # Backtesting command
    backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtesting')
    backtest_parser.add_argument('--strategy', required=True,
                               choices=['momentum', 'mean_reversion', 'ml'],
                               help='Trading strategy to test')
    backtest_parser.add_argument('--data', type=Path, required=True,
                               help='Path to historical data')
    backtest_parser.add_argument('--capital', type=float, default=100000.0,
                               help='Initial capital')

    # Web API command
    web_parser = subparsers.add_parser('web', help='Start web API server')
    web_parser.add_argument('--host', default='0.0.0.0',
                          help='Host address to bind')
    web_parser.add_argument('--port', type=int, default=8000,
                          help='Port number to listen')
    
    return parser

def run_data_pipeline(args, config: Dict) -> None:
    """Execute data pipeline operations"""
    logger.info("Starting data pipeline...")
    
    if args.data_command == 'ingest':
        logger.info(f"Ingesting data from {args.source}")
        ingestor = DataIngestor()
        
        # Load source-specific configuration
        source_config = config.get('data_sources', {}).get(args.source, {})
        
        # Ingest data
        for symbol in args.symbols.split(','):
            data = ingestor.ingest_data(
                symbol=symbol.strip(),
                source=args.source,
                **source_config
            )
            output_file = args.output / f"{symbol}.parquet"
            data.to_parquet(output_file)
            logger.info(f"Saved {symbol} data to {output_file}")
            
    elif args.data_command == 'preprocess':
        logger.info(f"Preprocessing data from {args.input}")
        preprocessor = DataPreprocessor()
        
        for data_file in args.input.glob('*.parquet'):
            raw_data = pd.read_parquet(data_file)
            processed_data = preprocessor.run_pipeline(raw_data)
            output_file = args.output / data_file.name
            processed_data.to_parquet(output_file)
            logger.info(f"Processed {data_file} -> {output_file}")

def run_rl_training(args, config: Dict) -> None:
    """Execute reinforcement learning operations"""
    logger.info("Starting RL training...")
    
    # Load environment configuration
    env_config = config.get('rl_environment', {})
    
    # Initialize environment
    data = pd.read_parquet(args.data)
    env = PortfolioEnv(data, **env_config)
    
    # Initialize agent
    agent_config = config.get('rl_agents', {}).get(args.agent, {})
    
    if args.agent == 'dqn':
        from reinforcement_learning.agents.dqn_agent import DQNAgent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **agent_config
        )
    elif args.agent == 'ppo':
        from reinforcement_learning.agents.ppo_agent import PPOAgent
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **agent_config
        )
    
    # Initialize trainer
    trainer = Trainer(env=env, agent=agent, **config.get('rl_training', {}))
    
    if args.rl_command == 'train':
        logger.info(f"Training {args.agent} agent for {args.episodes} episodes")
        trainer.train(n_episodes=args.episodes)
        model_path = Path(f"models/{args.agent}_model.pt")
        trainer.save_checkpoint(model_path)
        logger.info(f"Saved trained model to {model_path}")
        
    elif args.rl_command == 'evaluate':
        logger.info(f"Evaluating model from {args.model}")
        trainer.load_checkpoint(args.model)
        results = trainer.evaluate(n_episodes=args.episodes)
        logger.info(f"Evaluation results: {results}")

def run_backtesting(args, config: Dict) -> None:
    """Execute backtesting operation"""
    logger.info(f"Backtesting {args.strategy} strategy...")
    
    data = pd.read_parquet(args.data)
    backtester = Backtester(data, initial_capital=args.capital)
    results = backtester.run_backtest()
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_equity_curve(backtester.results, save_path='results/equity_curve.png')
    
    logger.info("Backtest results:")
    for metric, value in results.items():
        logger.info(f"{metric:>20}: {value:.2f}")

def run_web_server(args, config: Dict) -> None:
    """Start web API server"""
    from uvicorn import Config, Server
    
    server_config = config.get('web_server', {})
    ssl_config = server_config.get('ssl', {})
    
    server = Server(
        Config(
            web_app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=ssl_config.get('keyfile'),
            ssl_certfile=ssl_config.get('certfile')
        )
    )
    
    logger.info(f"Starting web server on {args.host}:{args.port}")
    server.run()

def main():
    """Main application entry point"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'data':
            run_data_pipeline(args, config)
        elif args.command == 'rl':
            run_rl_training(args, config)
        elif args.command == 'backtest':
            run_backtesting(args, config)
        elif args.command == 'web':
            run_web_server(args, config)
            
        logger.info("Operation completed successfully")
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
