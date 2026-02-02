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
    format="%(message)s",
    handlers=[
        # This replaces the 'filename' and 'filemode' arguments
        logging.FileHandler("app.log", mode="a"),
        # This handles the terminal output
        logging.StreamHandler(sys.stdout),
    ],
    force=True,  # Ensures this config is applied even if OpenSpiel set one up already
)


gomuko = GomukoGame()

state = gomuko.new_initial_state()

# # Print the initial state
print(str(state))

env = rl_environment.Environment(gomuko)
state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
hidden_layers_sizes = [32, 32]
replay_buffer_capacity = int(1e4)

dqn_agent = DQN(
    player_id=0,
    state_representation_size=state_size,
    num_actions=num_actions,
    hidden_layers_sizes=hidden_layers_sizes,
    replay_buffer_capacity=replay_buffer_capacity,
)


def pretty_board(time_step):
    """Returns the 7x7 board with correct plane mapping."""
    info_state = np.array(time_step.observations["info_state"][0])
    board_size = 7
    num_cells = board_size * board_size

    # Plane 1 is actually Player 1 (X)
    x_start, x_end = num_cells, 2 * num_cells
    x_locations = np.nonzero(info_state[x_start:x_end])[0]

    # Plane 2 is actually Player 2 (O)
    o_start, o_end = 2 * num_cells, 3 * num_cells
    o_locations = np.nonzero(info_state[o_start:o_end])[0]

    board = np.full(num_cells, ".")
    board[x_locations] = "X"
    board[o_locations] = "O"

    return board.reshape((board_size, board_size))


def command_line_action(time_step):
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


dqn_agent.load("gomuko/checkpoints", False)
human_player = 1

while True:
    logging.info("You are playing as %s", "X" if human_player else "0")
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]

        if player_id == human_player:
            # agent_out = dqn_agent.step(time_step, is_evaluation=True)
            # logging.info("\n%s", agent_out.probs.reshape((7, 7)))
            action = command_line_action(time_step)
        else:
            agent_out = dqn_agent.step(time_step, is_evaluation=True)
            action = agent_out.action

        time_step = env.step([action])
        logging.info("\n%s", pretty_board(time_step))

    logging.info("\n%s", pretty_board(time_step))

    logging.info("End of game!")
    if time_step.rewards[human_player] > 0:
        logging.info("You win")
    elif time_step.rewards[human_player] < 0:
        logging.info("You lose")
    else:
        logging.info("Draw")
