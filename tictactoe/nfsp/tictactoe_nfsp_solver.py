"""NFSP TicTacToe solver with ELO rating and random bot evaluation.

Uses Neural Fictitious Self-Play to learn to play TicTacToe.
Includes evaluation against random bots and ELO tracking.
"""

from typing import List, Tuple
import os
import sys


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
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy
from algorithms import nfsp


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
        """Update rating based on match results."""
        total_games = wins + losses + draws
        if total_games == 0:
            return self.current_rating

        actual_score = (wins + 0.5 * draws) / total_games
        expected = self.expected_score(self.current_rating, opponent_rating)

        self.current_rating += self.k_factor * (actual_score - expected)
        self.rating_history.append(self.current_rating)
        return self.current_rating


class HistoricalOpponentPool:
    """Maintains a pool of past NFSP agent network weights with ELO ratings.

    Instead of deepcopy (which fails on PyTorch tensors with gradients),
    we store the state_dicts of the networks.
    """

    def __init__(self, max_size: int = 10, initial_elo: float = 1000.0):
        self.max_size = max_size
        # Store (avg_network_state_dicts, rl_agent_q_network_state_dicts, elo_rating)
        self.pool: List[Tuple[List[dict], List[dict], float]] = []
        self.initial_elo = initial_elo

    def add_snapshot(self, agents: List[nfsp.NFSP], elo_rating: float):
        """Save copies of network weights from current agents."""
        avg_states = [
            {k: v.clone().detach() for k, v in agent._avg_network.state_dict().items()}
            for agent in agents
        ]
        rl_states = [
            {
                k: v.clone().detach()
                for k, v in agent._rl_agent._q_network.state_dict().items()
            }
            for agent in agents
        ]
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append((avg_states, rl_states, elo_rating))

    def load_into_agents(self, agents: List[nfsp.NFSP], idx: int):
        """Load snapshot weights into given agents for evaluation."""
        avg_states, rl_states, _ = self.pool[idx]
        for i, agent in enumerate(agents):
            agent._avg_network.load_state_dict(avg_states[i])
            agent._rl_agent._q_network.load_state_dict(rl_states[i])

    def get_elo(self, idx: int) -> float:
        """Get ELO rating for snapshot at index."""
        return self.pool[idx][2]

    def get_latest_elo(self) -> float:
        """Get ELO rating of most recent snapshot."""
        if not self.pool:
            return self.initial_elo
        return self.pool[-1][2]

    def __len__(self):
        return len(self.pool)


def play_match_nfsp(
    env: rl_environment.Environment,
    agents1: List[nfsp.NFSP],
    agents2,  # Can be List[nfsp.NFSP] or List[RandomAgent]
    num_games: int,
    agents1_player: int = 0,  # Which player position agents1 plays
) -> Tuple[int, int, int]:
    """Play a match between NFSP agents.

    Returns (wins, losses, draws) from agents1's perspective.
    """
    wins = losses = draws = 0

    for _ in range(num_games):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]

            if player_id == agents1_player:
                with agents1[player_id].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                    agent_output = agents1[player_id].step(
                        time_step, is_evaluation=True
                    )
            else:
                if isinstance(agents2[player_id], nfsp.NFSP):
                    with agents2[player_id].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                        agent_output = agents2[player_id].step(
                            time_step, is_evaluation=True
                        )
                else:
                    agent_output = agents2[player_id].step(
                        time_step, is_evaluation=True
                    )

            time_step = env.step([agent_output.action])

        if time_step.rewards[agents1_player] > 0:
            wins += 1
        elif time_step.rewards[agents1_player] < 0:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


def eval_against_historical_nfsp(
    env: rl_environment.Environment,
    agents: List[nfsp.NFSP],
    opponents: List[nfsp.NFSP],
    num_games_per_side: int = 100,
) -> Tuple[int, int, int]:
    """Evaluate NFSP agents against historical opponents, playing both sides."""
    # Play as P0
    w1, l1, d1 = play_match_nfsp(
        env, agents, opponents, num_games_per_side, agents1_player=0
    )
    print(f"Play as P0: W{w1}/L{l1}/D{d1}")
    # Play as P1
    w2, l2, d2 = play_match_nfsp(
        env, agents, opponents, num_games_per_side, agents1_player=1
    )
    print(f"Play as P1: W{w2}/L{l2}/D{d2}")

    return w1 + w2, l1 + l2, d1 + d2


def eval_against_random_bots(
    env: rl_environment.Environment,
    agents: List[nfsp.NFSP],
    random_agents: List[random_agent.RandomAgent],
    num_episodes: int,
) -> np.ndarray:
    """Evaluates NFSP agents playing as P0 and P1 against random bots."""
    wins = np.zeros(2)
    for player_pos in range(2):
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == player_pos:
                    with agents[player_id].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                        agent_output = agents[player_id].step(
                            time_step, is_evaluation=True
                        )
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
    num_train_episodes = 800000
    eval_interval = 200000
    save_model_interval = 100000
    seed = 42

    # NFSP hyperparameters
    hidden_layers_sizes = [128]
    replay_buffer_capacity = int(2e5)
    reservoir_buffer_capacity = int(2e6)
    anticipatory_param = 0.1

    # Create environment
    env = Environment(
        TicTacToeGame(), chance_event_sampler=ChanceEventSampler(seed=seed)
    )

    # Get environment specs
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = 2

    print(f"Info state size: {info_state_size}")
    print(f"Num actions: {num_actions}")

    # Create NFSP agents
    kwargs = {
        "replay_buffer_capacity": replay_buffer_capacity,
        "epsilon_decay_duration": num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    agents = [
        nfsp.NFSP(
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            reservoir_buffer_capacity=reservoir_buffer_capacity,
            anticipatory_param=anticipatory_param,
            **kwargs,
        )
        for idx in range(num_players)
    ]

    # ELO rating tracker
    # ELO rating tracker (based on win rate vs random)
    best_winrate = 0.0

    # Random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    print("\nTraining NFSP agents")
    print(f"  - Hidden layers: {hidden_layers_sizes}")
    print(f"  - Anticipatory param: {anticipatory_param}")
    print(f"  - Replay buffer: {replay_buffer_capacity}")
    print(f"  - Reservoir buffer: {reservoir_buffer_capacity}")

    # Training loop
    for ep in trange(num_train_episodes, desc="Training"):
        # Evaluate periodically
        if ep and ep % eval_interval == 0:
            # Win rate vs random
            r_mean = eval_against_random_bots(env, agents, random_agents, 100)
            avg_winrate = (r_mean[0] + r_mean[1]) / 2

            if avg_winrate > best_winrate:
                best_winrate = avg_winrate
                improved = " (NEW BEST!)"
            else:
                improved = ""

            # # Exploitability
            # try:
            #     expl = exploitability.exploitability(env.game, expl_policies_avg)
            #     expl_str = f"Expl={expl:.4f}"
            # except Exception:
            #     expl_str = "Expl=N/A"

            # Losses
            losses_info = [agent.loss for agent in agents]

            print(
                f"\n[Ep {ep}] WR vs Random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}, "
                f"Avg={avg_winrate:.3f}{improved}"
            )
            print(f"  Losses: {losses_info}")

        # Training episode
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state
        for agent in agents:
            agent.step(time_step)

        if ep and ep % save_model_interval == 0:
            agents[0].save(f"tictactoe/nfsp/checkpoints/nfsp_p0_ep{ep}")
            agents[1].save(f"tictactoe/nfsp/checkpoints/nfsp_p1_ep{ep}")
            print(f"  Models saved at episode {ep}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    print(f"\nBest avg win rate: {best_winrate:.3f}")

    r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
    print(f"\nWin rate vs random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}")

    # Exploitability
    # try:
    #     expl = exploitability.exploitability(env.game, expl_policies_avg)
    #     print(f"Final exploitability: {expl:.6f}")
    # except Exception as e:
    #     print(f"Could not compute exploitability: {e}")

    # Self-play evaluation (average policy mode)
    print("\nNFSP vs NFSP - Average Policy (1000 games):")
    p0_wins = p1_wins = draws = 0
    for _ in range(1000):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            with agents[player_id].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                agent_output = agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
        if time_step.rewards[0] > 0:
            p0_wins += 1
        elif time_step.rewards[1] > 0:
            p1_wins += 1
        else:
            draws += 1
    print(f"P0 wins: {p0_wins}, P1 wins: {p1_wins}, Draws: {draws}")


if __name__ == "__main__":
    main()
