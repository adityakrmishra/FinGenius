    def save_checkpoint(self, path: str) -> None:
        """Save agent and environment state"""
        torch.save({
            'agent_state': self.agent.state_dict(),
            'env_current_step': self.env.current_step,
            'env_balance': self.env.balance,
            'env_portfolio': self.env.portfolio,
            'optimizer_state': self.agent.optimizer.state_dict(),
            'current_episode': self.current_episode,
            'episode_rewards': self.episode_rewards,
            'portfolio_values': self.portfolio_values,
            'epsilon': self.agent.epsilon if hasattr(self.agent, 'epsilon') else None,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent and environment state"""
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state'])
        self.env.current_step = checkpoint['env_current_step']
        self.env.balance = checkpoint['env_balance']
        self.env.portfolio = checkpoint['env_portfolio']
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_episode = checkpoint['current_episode']
        self.episode_rewards = checkpoint['episode_rewards']
        self.portfolio_values = checkpoint['portfolio_values']
        if hasattr(self.agent, 'epsilon') and 'epsilon' in checkpoint:
            self.agent.epsilon = checkpoint['epsilon']
