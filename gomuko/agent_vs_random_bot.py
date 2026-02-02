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

gomuko = GomukoGame()

env = rl_environment.Environment(gomuko)
state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

num_players = 2

hidden_layers_sizes = [32, 32]
replay_buffer_capacity = int(1e4)
train_episodes = 200000
loss_report_interval = 1000
save_model_interval = 5000


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
        for _ in trange(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


agents = [
    DQN(
        player_id=0,
        state_representation_size=state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=replay_buffer_capacity,
        device="cuda",
        epsilon_end=1.0,
    ),
    DQN(
        player_id=1,
        state_representation_size=state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=replay_buffer_capacity,
        device="cuda",
        epsilon_end=1.0,
    ),
]

agents[0].load("gomuko/checkpoints/agent_0_checkpoint_5000.pt", False)
agents[1].load("gomuko/checkpoints/agent_1_checkpoint_5000.pt", False)

random_agents = [
    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]

r_mean = eval_against_random_bots(env, agents, random_agents, 100)

print("Mean episode rewards: %s", r_mean)
