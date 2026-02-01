"""PPO vs PPO TicTacToe solver with parallel environments for faster sampling.

For turn-based games, we collect complete episodes and then train.
Each PPO agent collects experiences only when it's their turn.
"""

from typing import List
import os
import sys

# Add parent directory to path so we can import from algorithims
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from open_spiel.python.games.tic_tac_toe import TicTacToeGame
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.rl_environment import Environment, ChanceEventSampler
from open_spiel.python.algorithms import random_agent
from algorithims.ppo import PPO


def eval_against_random_bots(
    env: rl_environment.Environment,
    trained_agents: List[PPO],
    random_agents: List[random_agent.RandomAgent],
    num_episodes: int,
) -> np.ndarray:
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
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
    num_envs = 8  # Number of parallel environments (games)
    num_updates = 500  # Number of update cycles
    episodes_per_update = 128  # Episodes to collect before each update
    eval_interval = 25  # Evaluate every N updates
    seed = 42

    # Create multiple environments
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

    # Create two PPO agents - one for each player
    # Each agent uses num_envs=1 but we'll batch experiences manually
    agents = [
        PPO(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=num_players,
            player_id=player_id,
            num_envs=1,  # Single env interface, we batch manually
            steps_per_batch=128,
        )
        for player_id in range(num_players)
    ]

    # Random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # Single env for evaluation
    eval_env = envs[0]

    # Training loop
    for update in trange(num_updates, desc="Training"):
        # Evaluate periodically
        if update % eval_interval == 0:
            r_mean = eval_against_random_bots(eval_env, agents, random_agents, 500)
            print(
                f"\n[Update {update}] Win rate vs random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}"
            )

        # Collect episodes across all parallel envs
        episodes_collected = 0
        env_idx = 0

        while episodes_collected < episodes_per_update:
            env = envs[env_idx % num_envs]
            env_idx += 1

            time_step = env.reset()

            # Play one complete episode
            while not time_step.last():
                player_id = time_step.observations["current_player"]

                # Get action from current player's agent
                agent_output = agents[player_id].step([time_step])[0]
                prev_time_step = time_step
                time_step = env.step([agent_output.action])

                # Store experience for the agent that just acted
                reward = [time_step.rewards[player_id]]
                done = [time_step.last()]
                agents[player_id].post_step(reward, done)

                # Trigger learning when batch is full
                if agents[player_id].cur_batch_idx >= agents[player_id].steps_per_batch:
                    agents[player_id].learn([time_step])

            episodes_collected += 1

        # Ensure both agents learn at end of collection phase
        for player_id in range(num_players):
            if agents[player_id].cur_batch_idx > 0:
                # Need to learn with remaining experiences
                # Pad if necessary or learn with what we have
                if agents[player_id].cur_batch_idx >= agents[player_id].steps_per_batch:
                    agents[player_id].learn([time_step])

    # Final evaluation
    print("\n=== Final Evaluation ===")
    r_mean = eval_against_random_bots(eval_env, agents, random_agents, 1000)
    print(f"Final win rate vs random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}")

    # PPO vs PPO evaluation
    print("\nPPO vs PPO (1000 games):")
    p0_wins = 0
    p1_wins = 0
    draws = 0
    for _ in range(1000):
        time_step = eval_env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step([time_step], is_evaluation=True)[0]
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
