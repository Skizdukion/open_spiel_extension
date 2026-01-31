from typing import List
import os
import sys

# Add parent directory to path so we can import from algorithims
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from game.gomuko import GomukoGame
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
import logging

from algorithims.dqn import DQN

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
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

    hidden_layers_sizes = [512, 512]
    replay_buffer_capacity = int(1e4)
    train_episodes = 200000
    loss_report_interval = 1000
    save_model_interval = 5000

    agents = [
        DQN(
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity,
            device="cuda",
        ),
        DQN(
            player_id=1,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity,
            device="cuda",
        ),
    ]

    for ep in trange(train_episodes):
        if ep and ep % loss_report_interval == 0:
            logging.info("[%s/%s] DQN 1 loss: %s", ep, train_episodes, agents[0].loss)
            logging.info("[%s/%s] DQN 2 loss: %s", ep, train_episodes, agents[1].loss)

        if ep and ep % save_model_interval == 0:
            agents[0].save(f"gomuko/checkpoints/agent_{0}_checkpoint_{ep}.pt")
            agents[1].save(f"gomuko/checkpoints/agent_{1}_checkpoint_{ep}.pt")

        time_step = env.reset()

        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]
    r_mean = eval_against_random_bots(env, agents, random_agents, 1000)

    for i in range(agents):
        agents[i].save(f"gomuko/agent_{i}_checkpoint.pt")

    logging.info("Mean episode rewards: %s", r_mean)


main()

# tictactoe = TicTacToeGame()

# state = tictactoe.new_initial_state()

# # Print the initial state
# print(str(state))

# while not state.is_terminal():
#     # The state can be three different types: chance node,
#     # simultaneous node, or decision node
#     if state.is_chance_node():
#         # Chance node: sample an outcome
#         outcomes = state.chance_outcomes()
#         num_actions = len(outcomes)
#         print("Chance node, got " + str(num_actions) + " outcomes")
#         action_list, prob_list = zip(*outcomes)
#         action = np.random.choice(action_list, p=prob_list)
#         print(
#             "Sampled outcome: ", state.action_to_string(state.current_player(), action)
#         )
#         state.apply_action(action)

#     elif state.is_simultaneous_node():
#         # Simultaneous node: sample actions for all players.

#         def random_choice(a):
#             return np.random.choice(a) if a else [0]

#         chosen_actions = [
#             random_choice(state.legal_actions(pid))
#             for pid in range(tictactoe.num_players())
#         ]
#         print(
#             "Chosen actions: ",
#             [
#                 state.action_to_string(pid, action)
#                 for pid, action in enumerate(chosen_actions)
#             ],
#         )
#         state.apply_actions(chosen_actions)
#     else:
#         # Decision node: sample action for the single current player
#         action = random.choice(state.legal_actions(state.current_player()))
#         action_string = state.action_to_string(state.current_player(), action)
#         print(
#             "Player ",
#             state.current_player(),
#             ", randomly sampled action: ",
#             action_string,
#         )
#         state.apply_action(action)

#     print(str(state))

# # Game is now done. Print utilities for each player
# returns = state.returns()
# for pid in range(tictactoe.num_players()):
#     print("Utility for player {} is {}".format(pid, returns[pid]))
