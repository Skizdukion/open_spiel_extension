"""PPO vs PPO TicTacToe solver with separate agents and historical opponent pool.

Key features:
1. Two separate PPO agents - one for each player position
2. Historical opponent pool - train against past versions of opponents
3. Alternating training: each episode trains ONE agent against various opponents
"""

from typing import List
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
from algorithms.ppo import PPO


class HistoricalOpponentPool:
    """Maintains a pool of past agent snapshots for training."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: List[PPO] = []

    def add_snapshot(self, agent: PPO):
        """Add a copy of the current agent to the pool."""
        snapshot = copy.deepcopy(agent)
        snapshot.eval()
        if len(self.pool) >= self.max_size:
            self.pool.pop(0)
        self.pool.append(snapshot)

    def sample_opponent(self) -> PPO:
        """Sample a random opponent from the pool."""
        if not self.pool:
            return None
        return np.random.choice(self.pool)

    def __len__(self):
        return len(self.pool)


def eval_against_random_bots(
    env: rl_environment.Environment,
    agents: List[PPO],
    random_agents: List[random_agent.RandomAgent],
    num_episodes: int,
) -> np.ndarray:
    """Evaluates agents against random bots."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if isinstance(cur_agents[player_id], PPO):
                    agent_output = cur_agents[player_id].step(
                        [time_step], is_evaluation=True
                    )[0]
                else:
                    agent_output = cur_agents[player_id].step(
                        time_step, is_evaluation=True
                    )
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


def main():
    # Configuration
    num_envs = 1
    num_updates = 2000
    episodes_per_update = 64
    eval_interval = 25
    snapshot_interval = 200
    seed = 42

    # Opponent selection probabilities
    self_play_prob = 0.4  # 40% vs current opponent (both train)
    historical_prob = 0.4  # 40% vs historical
    random_prob = 0.2  # 20% vs random

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

    # Create TWO separate PPO agents
    agents = [
        PPO(
            input_shape=(info_state_shape,),
            num_actions=num_actions,
            num_players=num_players,
            player_id=player_id,
            num_envs=1,
            steps_per_batch=128,
        )
        for player_id in range(num_players)
    ]

    # Historical opponent pools
    opponent_pools = [HistoricalOpponentPool(max_size=10) for _ in range(num_players)]

    # Random agents
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    eval_env = envs[0]

    print("\nTraining with:")
    print(f"  - Self-play (both train): {self_play_prob*100:.0f}%")
    print(f"  - Historical opponents: {historical_prob*100:.0f}%")
    print(f"  - Random opponents: {random_prob*100:.0f}%")

    # Training loop
    for update in trange(num_updates, desc="Training"):
        # Evaluate periodically
        if update % eval_interval == 0:
            r_mean = eval_against_random_bots(eval_env, agents, random_agents, 500)
            pool_sizes = [len(p) for p in opponent_pools]
            print(
                f"\n[Update {update}] Win rate vs random: "
                f"P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}, "
                f"Pool sizes={pool_sizes}"
            )

        # Save snapshots periodically
        if update > 0 and update % snapshot_interval == 0:
            for player_id in range(num_players):
                opponent_pools[player_id].add_snapshot(agents[player_id])
            print("Added snapshots to pools")

        # Collect episodes
        episodes_collected = 0
        env_idx = 0

        while episodes_collected < episodes_per_update:
            env = envs[env_idx % num_envs]
            env_idx += 1

            time_step = env.reset()

            # Decide training setup for this episode
            roll = np.random.random()

            if roll < self_play_prob:
                # Self-play: BOTH agents train against each other
                # training_mode = "self_play"
                game_agents = [agents[0], agents[1]]
                training_players = [0, 1]  # Both train
            else:
                # Single agent training: pick one to train
                training_player = np.random.randint(0, 2)
                opponent_player = 1 - training_player
                training_players = [training_player]  # Only one trains

                if (
                    roll < self_play_prob + historical_prob
                    and len(opponent_pools[opponent_player]) > 0
                ):
                    # vs Historical
                    # training_mode = "historical"
                    opponent = opponent_pools[opponent_player].sample_opponent()
                else:
                    # vs Random
                    # training_mode = "random"
                    opponent = random_agents[opponent_player]

                # Build game_agents list
                game_agents = [None, None]
                game_agents[training_player] = agents[training_player]
                game_agents[opponent_player] = opponent

            # Play one complete episode
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                current_agent = game_agents[player_id]

                # Get action from current agent
                if isinstance(current_agent, PPO):
                    # Determine if this agent is training or just acting
                    is_training = player_id in training_players
                    if is_training:
                        agent_output = current_agent.step([time_step])[0]
                    else:
                        agent_output = current_agent.step(
                            [time_step], is_evaluation=True
                        )[0]
                else:
                    # Random agent
                    agent_output = current_agent.step(time_step, is_evaluation=True)

                time_step = env.step([agent_output.action])

                # Store experience only for training agents
                if player_id in training_players:
                    reward = [time_step.rewards[player_id]]
                    done = [time_step.last()]
                    agents[player_id].post_step(reward, done)

                    # Trigger learning when batch is full
                    if (
                        agents[player_id].cur_batch_idx
                        >= agents[player_id].steps_per_batch
                    ):
                        agents[player_id].learn([time_step])

            episodes_collected += 1

    # Final evaluation
    print("\n=== Final Evaluation ===")
    r_mean = eval_against_random_bots(eval_env, agents, random_agents, 1000)
    print(f"Final win rate vs random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}")

    # # PPO vs PPO - Deterministic (greedy)
    # print("\nPPO vs PPO - Deterministic (1000 games):")
    # p0_wins = 0
    # p1_wins = 0
    # draws = 0
    # for _ in range(1000):
    #     time_step = eval_env.reset()
    #     while not time_step.last():
    #         player_id = time_step.observations["current_player"]
    #         agent_output = agents[player_id].step([time_step], is_evaluation=True)[0]
    #         time_step = eval_env.step([agent_output.action])
    #     if time_step.rewards[0] > 0:
    #         p0_wins += 1
    #     elif time_step.rewards[1] > 0:
    #         p1_wins += 1
    #     else:
    #         draws += 1
    # print(f"P0 wins: {p0_wins}, P1 wins: {p1_wins}, Draws: {draws}")

    # PPO vs PPO - Stochastic (sample from policy)
    print("\nPPO vs PPO - Stochastic (1000 games):")
    p0_wins = 0
    p1_wins = 0
    draws = 0
    for _ in range(1000):
        time_step = eval_env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            # Use training mode (stochastic) but don't store experiences
            agent_output = agents[player_id].step([time_step], is_evaluation=False)[0]
            # Reset the batch index since we don't want to actually train
            agents[player_id].cur_batch_idx = max(
                0, agents[player_id].cur_batch_idx - 1
            )
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
