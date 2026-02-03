import numpy as np
from open_spiel.python import rl_agent
import copy


class HistoricalOpponentPool:
    """Maintains a pool of past agent snapshots for training."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: list[rl_agent] = []

    def add_snapshot(self, agent: rl_agent):
        """Add a copy of the current agent to the pool."""
        snapshot = copy.deepcopy(agent)
        snapshot.eval()
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

