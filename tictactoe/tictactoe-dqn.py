from typing import List
import importlib.util
import os

from tqdm import trange
from open_spiel.python.games.tic_tac_toe import TicTacToeGame, TicTacToeState
import numpy as np
import random
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
import sys
import logging
from explore.pytorch.dqn import DQN

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def pretty_board(time_step: rl_environment.TimeStep) -> np.ndarray:
    """Returns the board in `time_step` in a human readable format."""
    info_state = time_step.observations["info_state"][0]
    x_locations = np.nonzero(info_state[9:18])[0]
    o_locations = np.nonzero(info_state[18:])[0]
    board = np.full(3 * 3, ".")
    board[x_locations] = "X"
    board[o_locations] = "0"
    board = np.reshape(board, (3, 3))
    return board


def command_line_action(time_step: rl_environment.TimeStep) -> int:
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("Choose an action from {}:".format(legal_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action


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
    tictactoe = TicTacToeGame()

    env = rl_environment.Environment(tictactoe)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    num_players = 2

    hidden_layers_sizes = [32, 32]
    replay_buffer_capacity = int(1e4)
    train_episodes = 10000
    loss_report_interval = 1000

    dqn_agent = DQN(
        player_id=0,
        state_representation_size=state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=replay_buffer_capacity,
    )

    tabular_q_agent = tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)
    agents = [dqn_agent, tabular_q_agent]

    for ep in trange(train_episodes):
        if ep and ep % loss_report_interval == 0:
            logging.info("[%s/%s] DQN loss: %s", ep, train_episodes, agents[0].loss)

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
