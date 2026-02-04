import numpy as np
from open_spiel.python import rl_agent
import copy


def clone_dqn_agent(agent):
    """Create a clone of a DQN agent without using deepcopy.

    This avoids the RuntimeError that occurs when deepcopying PyTorch
    tensors with autograd history.
    """
    # Use stored kwargs to create a new instance with same hyperparameters
    kwargs = agent._kwargs.copy()
    # Remove 'self' from kwargs as it was captured by locals()
    kwargs.pop("self", None)

    # Create new agent instance
    agent_class = type(agent)
    new_agent = agent_class(**kwargs)

    # Copy network weights
    new_agent._q_network.load_state_dict(agent._q_network.state_dict())
    new_agent._target_q_network.load_state_dict(agent._target_q_network.state_dict())

    # Copy iteration counter
    new_agent._iteration = agent._iteration

    return new_agent


class HistoricalOpponentPool:
    """Maintains a pool of past agent snapshots for training."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: list[rl_agent] = []

    def add_snapshot(self, agent: rl_agent):
        """Add a copy of the current agent to the pool."""
        # Use clone_dqn_agent for PyTorch-based agents to avoid deepcopy issues
        if hasattr(agent, "_kwargs") and hasattr(agent, "_q_network"):
            snapshot = clone_dqn_agent(agent)
        else:
            snapshot = copy.deepcopy(agent)

        if hasattr(snapshot, "eval"):
            snapshot.eval()
        elif hasattr(snapshot, "_q_network"):
            snapshot._q_network.eval()
            if hasattr(snapshot, "_target_q_network"):
                snapshot._target_q_network.eval()
                
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append(snapshot)

    def sample_opponent(self) -> rl_agent:
        """Sample a random opponent from the pool."""
        if not self.pool:
            return None
        return np.random.choice(self.pool)

    def __len__(self):
        return len(self.pool)
