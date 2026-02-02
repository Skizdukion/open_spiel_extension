"""PPO TicTacToe solver with SHARED NETWORK for both positions.

Uses PPOMultiPosition to train a single network that plays as both P0 and P1.
Includes ELO rating tracking against past versions of itself.
"""

from typing import List, Tuple
import os
import sys
import copy

# Add parent directory to path so we can import from algorithms
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tqdm import trange
from game.tictactoe import TicTacToeGame
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.rl_environment import Environment, ChanceEventSampler
from open_spiel.python.algorithms import random_agent
from algorithms.ppo_multi_position import PPOMultiPosition
from util.canical_obs import tictactoe_canonical_obs


class EloRating:
    """ELO rating tracker for agents."""

    def __init__(self, initial_rating: float = 1000.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.current_rating = initial_rating
        self.rating_history = [initial_rating]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_rating(
        self, wins: int, losses: int, draws: int, opponent_rating: float
    ) -> float:
        """Update rating based on match results.

        Returns the new rating.
        """
        total_games = wins + losses + draws
        if total_games == 0:
            return self.current_rating

        # Actual score: win=1, draw=0.5, loss=0
        actual_score = (wins + 0.5 * draws) / total_games
        expected = self.expected_score(self.current_rating, opponent_rating)

        # Update rating
        self.current_rating += self.k_factor * (actual_score - expected)
        self.rating_history.append(self.current_rating)
        return self.current_rating


class HistoricalOpponentPool:
    """Maintains a pool of past agent snapshots with ELO ratings."""

    def __init__(self, max_size: int = 10, initial_elo: float = 1000.0):
        self.max_size = max_size
        self.pool: List[Tuple[PPOMultiPosition, float]] = []  # (agent, elo_rating)
        self.initial_elo = initial_elo

    def add_snapshot(self, agent: PPOMultiPosition, elo_rating: float):
        """Add a copy of the current agent to the pool with its rating."""
        snapshot = copy.deepcopy(agent)
        snapshot.eval()
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append((snapshot, elo_rating))

    def sample_opponent(self) -> Tuple[PPOMultiPosition, float]:
        """Sample a random opponent from the pool. Returns (agent, elo_rating)."""
        if not self.pool:
            return None, self.initial_elo
        idx = np.random.randint(len(self.pool))
        return self.pool[idx]

    def get_latest(self) -> Tuple[PPOMultiPosition, float]:
        """Get the most recent historical agent."""
        if not self.pool:
            return None, self.initial_elo
        return self.pool[-1]

    def __len__(self):
        return len(self.pool)


def play_match(
    env: rl_environment.Environment,
    agent1: PPOMultiPosition,
    agent2,  # Can be PPOMultiPosition or RandomAgent
    num_games: int,
    agent1_as_p0: bool = True,
    stochastic: bool = False,  # If True, sample from policy instead of greedy
) -> Tuple[int, int, int]:
    """Play a match between two agents.

    Args:
        stochastic: If True, sample actions from policy distribution.
                   If False (default), use greedy/deterministic actions.

    Returns (wins, losses, draws) from agent1's perspective.
    """
    wins = losses = draws = 0

    for _ in range(num_games):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]

            # Determine which agent acts
            if (player_id == 0 and agent1_as_p0) or (
                player_id == 1 and not agent1_as_p0
            ):
                if stochastic:
                    # Sample from policy (non-deterministic)
                    agent_output = agent1.step([time_step], is_evaluation=False)[0]
                    # Undo buffer increment since we're just evaluating
                    agent1.cur_batch_idx[player_id] = max(
                        0, agent1.cur_batch_idx[player_id] - 1
                    )
                else:
                    agent_output = agent1.step([time_step], is_evaluation=True)[0]
            else:
                if isinstance(agent2, PPOMultiPosition):
                    if stochastic:
                        agent_output = agent2.step([time_step], is_evaluation=False)[0]
                        agent2.cur_batch_idx[player_id] = max(
                            0, agent2.cur_batch_idx[player_id] - 1
                        )
                    else:
                        agent_output = agent2.step([time_step], is_evaluation=True)[0]
                else:
                    agent_output = agent2.step(time_step, is_evaluation=True)

            time_step = env.step([agent_output.action])

        # Determine result from agent1's perspective
        agent1_player = 0 if agent1_as_p0 else 1
        if time_step.rewards[agent1_player] > 0:
            wins += 1
        elif time_step.rewards[agent1_player] < 0:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


def eval_against_historical(
    env: rl_environment.Environment,
    agent: PPOMultiPosition,
    opponent: PPOMultiPosition,
    num_games_per_side: int = 100,
    stochastic: bool = True,  # Use stochastic by default for realistic eval
) -> Tuple[int, int, int]:
    """Evaluate agent against historical opponent, playing both sides.

    Returns total (wins, losses, draws).
    """
    # Play as P0
    w1, l1, d1 = play_match(
        env,
        agent,
        opponent,
        num_games_per_side,
        agent1_as_p0=True,
        stochastic=stochastic,
    )
    print(f"Play as P0: W{w1}/L{l1}/D{d1}")
    # Play as P1
    w2, l2, d2 = play_match(
        env,
        agent,
        opponent,
        num_games_per_side,
        agent1_as_p0=False,
        stochastic=stochastic,
    )
    print(f"Play as P1: W{w2}/L{l2}/D{d2}")

    return w1 + w2, l1 + l2, d1 + d2


def eval_against_random_bots(
    env: rl_environment.Environment,
    agent: PPOMultiPosition,
    random_agents: List[random_agent.RandomAgent],
    num_episodes: int,
) -> np.ndarray:
    """Evaluates shared agent playing as P0 and P1 against random bots."""
    wins = np.zeros(2)
    for player_pos in range(2):
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == player_pos:
                    agent_output = agent.step([time_step], is_evaluation=True)[0]
                else:
                    agent_output = random_agents[player_id].step(
                        time_step, is_evaluation=True
                    )
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


def main():
    # Configuration
    num_envs = 1
    num_updates = 1000
    episodes_per_update = 64
    eval_interval = 50
    snapshot_interval = 100
    elo_eval_games = 100  # Games per side for ELO evaluation
    seed = 42

    # Opponent selection probabilities (will change during training)
    # Phase 1 (0-500): more random exploration
    # Phase 2 (500+): focus on self-play and historical
    phase_transition_update = 500

    # Create environments
    envs = [
        Environment(
            TicTacToeGame(), chance_event_sampler=ChanceEventSampler(seed=seed + i)
        )
        for i in range(num_envs)
    ]

    # Get environment specs
    info_state_shape = envs[0].observation_spec()["info_state"][0]
    num_actions = envs[0].action_spec()["num_actions"]
    num_players = 2

    print(f"Info state shape: {info_state_shape}")
    print(f"Num actions: {num_actions}")
    print(f"Num parallel envs: {num_envs}")

    # Create SINGLE shared PPO agent for BOTH positions
    agent = PPOMultiPosition(
        input_shape=(info_state_shape,),
        num_actions=num_actions,
        num_players=num_players,
        player_ids=[0, 1],
        num_envs=1,
        steps_per_batch=128,
        entropy_coef=0.05,  # Promote exploration
        canonical_obs_fn=tictactoe_canonical_obs,  # Enable canonical form
    )

    # ELO rating tracker
    elo = EloRating(initial_rating=1000.0, k_factor=32.0)

    # Historical opponent pool with ELO ratings
    opponent_pool = HistoricalOpponentPool(max_size=10, initial_elo=1000.0)

    # Random agents
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    eval_env = envs[0]

    print("\nTraining SHARED NETWORK for both positions with ELO tracking")
    print("  Curriculum:")
    print("    Phase 1 (0-500): 50% self, 30% historical, 20% random")
    print("    Phase 2 (500+):  70% self, 30% historical, 0% random")

    # Training loop
    for update in trange(num_updates, desc="Training"):
        # Evaluate periodically
        if update % eval_interval == 0:
            r_mean = eval_against_random_bots(eval_env, agent, random_agents, 500)

            # ELO evaluation against last historical snapshot
            if len(opponent_pool) > 0:
                hist_agent, hist_elo = opponent_pool.get_latest()
                wins, losses, draws = eval_against_historical(
                    eval_env, agent, hist_agent, elo_eval_games
                )
                new_elo = elo.update_rating(wins, losses, draws, hist_elo)
                elo_str = (
                    f"ELO={new_elo:.0f} (vs {hist_elo:.0f}: W{wins}/L{losses}/D{draws})"
                )
            else:
                elo_str = f"ELO={elo.current_rating:.0f} (no history yet)"

            print(
                f"\n[Update {update}] {elo_str}, "
                f"WR vs Random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}, "
                f"Pool={len(opponent_pool)}"
            )

        # Save snapshot periodically
        if update > 0 and update % snapshot_interval == 0:
            opponent_pool.add_snapshot(agent, elo.current_rating)
            print(
                f"  Snapshot saved (ELO={elo.current_rating:.0f}, pool size={len(opponent_pool)})"
            )

        # Collect episodes
        episodes_collected = 0
        env_idx = 0

        while episodes_collected < episodes_per_update:
            env = envs[env_idx % num_envs]
            env_idx += 1

            time_step = env.reset()

            # Dynamic opponent selection based on training phase
            if update < phase_transition_update:
                # Phase 1: more exploration with random
                self_play_prob = 0.5
                historical_prob = 0.3
                random_prob = 0.2
            else:
                # Phase 2: focus on self-play and historical
                self_play_prob = 0.7
                historical_prob = 0.3
                random_prob = 0.0

            # Decide opponent type for this episode
            roll = np.random.random()
            if roll < self_play_prob:
                opponent_type = "self"
                opponent = None
            elif roll < self_play_prob + historical_prob and len(opponent_pool) > 0:
                opponent_type = "historical"
                opponent, _ = opponent_pool.sample_opponent()
            else:
                opponent_type = "random"
                opponent = None

            if opponent_type != "self":
                training_player = np.random.randint(0, 2)
            else:
                training_player = None

            # Play one complete episode
            while not time_step.last():
                player_id = time_step.observations["current_player"]

                if opponent_type == "self":
                    agent_output = agent.step([time_step])[0]
                    acting_agent = agent
                elif player_id == training_player:
                    agent_output = agent.step([time_step])[0]
                    acting_agent = agent
                else:
                    if opponent_type == "historical":
                        agent_output = opponent.step([time_step], is_evaluation=True)[0]
                        acting_agent = None
                    else:
                        agent_output = random_agents[player_id].step(
                            time_step, is_evaluation=True
                        )
                        acting_agent = None

                prev_time_step = time_step
                time_step = env.step([agent_output.action])

                if acting_agent is agent:
                    reward = [time_step.rewards[player_id]]
                    done = [time_step.last()]
                    agent.post_step(prev_time_step, reward, done)

                if agent.should_learn():
                    agent.learn([time_step])

            episodes_collected += 1

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    print(f"\nFinal ELO Rating: {elo.current_rating:.0f}")
    print(f"ELO History: {[f'{r:.0f}' for r in elo.rating_history]}")

    r_mean = eval_against_random_bots(eval_env, agent, random_agents, 1000)
    print(f"\nWin rate vs random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}")

    # Self-play evaluation
    print("\nSelf-play - Stochastic (1000 games):")
    p0_wins = p1_wins = draws = 0
    for _ in range(1000):
        time_step = eval_env.reset()
        while not time_step.last():
            agent_output = agent.step([time_step], is_evaluation=False)[0]
            player_id = time_step.observations["current_player"]
            agent.cur_batch_idx[player_id] = max(0, agent.cur_batch_idx[player_id] - 1)
            time_step = eval_env.step([agent_output.action])
        if time_step.rewards[0] > 0:
            p0_wins += 1
        elif time_step.rewards[1] > 0:
            p1_wins += 1
        else:
            draws += 1
    print(f"P0 wins: {p0_wins}, P1 wins: {p1_wins}, Draws: {draws}")


if __name__ == "__main__":
    main()
