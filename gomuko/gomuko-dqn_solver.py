from typing import List
import os
import sys

# Add parent directory to path so we can import from algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from game.gomuko import GomukoGame
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
import logging

from algorithms.dqn import DQN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="a"),
        logging.StreamHandler(),  # Console output
    ],
)


def eval_against_random_bots(
    env: rl_environment.Environment,
    trained_agents: List[tabular_qlearner.QLearner],
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
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


def main():
    gomuko = GomukoGame()

    env = rl_environment.Environment(gomuko)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    num_players = 2

    hidden_layers_sizes = [256, 256]
    replay_buffer_capacity = int(1e6)
    train_episodes = 200000
    loss_report_interval = 1000
    save_model_interval = 10000
    eval_interval = 10000

    agents = [
        DQN(
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity,
            batch_size=1024,
            gradient_clipping=1.0,
            learning_rate=0.001,
            learn_every=100,
            device="cuda",
        ),
        DQN(
            player_id=1,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity,
            batch_size=1024,
            gradient_clipping=1.0,
            learning_rate=0.001,
            learn_every=100,
            device="cuda",
        ),
    ]

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    best_winrate = 0.0

    for ep in trange(train_episodes):
        # Periodic evaluation
        if ep and ep % eval_interval == 0:
            # Agent vs Random
            r_mean = eval_against_random_bots(env, agents, random_agents, 100)
            avg_winrate = (r_mean[0] + r_mean[1]) / 2
            improved = " (NEW BEST!)" if avg_winrate > best_winrate else ""
            if avg_winrate > best_winrate:
                best_winrate = avg_winrate

            # Agent vs Agent (stochastic)
            p0_wins = p1_wins = draws = 0
            for _ in range(100):
                time_step = env.reset()
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    agent_output = agents[player_id].step(time_step, is_evaluation=True)
                    action = np.random.choice(
                        len(agent_output.probs), p=agent_output.probs
                    )
                    time_step = env.step([action])
                if time_step.rewards[0] > 0:
                    p0_wins += 1
                elif time_step.rewards[1] > 0:
                    p1_wins += 1
                else:
                    draws += 1

            logging.info(
                f"\n[Ep {ep}] WR vs Random: P0={r_mean[0]:.3f}, P1={r_mean[1]:.3f}, Avg={avg_winrate:.3f}{improved}"
            )
            logging.info(f"  DQN vs DQN: P0={p0_wins}, P1={p1_wins}, Draws={draws}")
            logging.info(
                "[%s] WR vs Random: P0=%.3f, P1=%.3f", ep, r_mean[0], r_mean[1]
            )

        if ep and ep % loss_report_interval == 0:
            # Log loss and Q-value magnitude
            q1 = agents[0].q_values
            q2 = agents[1].q_values
            q1_mean = q1.mean().item() if q1 is not None else 0
            q2_mean = q2.mean().item() if q2 is not None else 0
            logging.info(
                f"  Loss: P0={agents[0].loss:.4f}, P1={agents[1].loss:.4f} | Q-mean: P0={q1_mean:.4f}, P1={q2_mean:.4f}"
            )

        if ep and ep % save_model_interval == 0:
            agents[0].save(f"gomuko/checkpoints/agent_{0}_checkpoint_{ep}.pt")
            agents[1].save(f"gomuko/checkpoints/agent_{1}_checkpoint_{ep}.pt")
            logging.info(f"  Models saved at episode {ep}")

        time_step = env.reset()

        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


main()
