from typing import List
import os
import sys

# Add parent directory to path so we can import from algorithims
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from open_spiel.python.games.tic_tac_toe import TicTacToeGame
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from algorithims.ppo import PPO
from open_spiel.python.algorithms import tabular_qlearner


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
    tictactoe = TicTacToeGame()

    env = rl_environment.Environment(tictactoe)
    # state_size = env.observation_spec()["info_state"][0]
    info_state_shape = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    random_bot_eval_interval = 10000

    num_players = 2

    train_episodes = 50000

    ppo_agent = PPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=2,
        player_id=0,
        num_envs=1,
    )

    tabular_q_agent = tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)
    agents = [ppo_agent, tabular_q_agent]

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    for ep in trange(train_episodes):
        if ep % random_bot_eval_interval == 0:
            r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
            print(f"Mean episode rewards: {r_mean}")

        time_step = env.reset()

        while not time_step.last():
            player_id = time_step.observations["current_player"]
            # PPO expects a list of TimeStep objects (for vectorized envs)
            if isinstance(agents[player_id], PPO):
                agent_output = agents[player_id].step([time_step])[0]
            else:
                agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

            # PPO needs post_step to store rewards and done flags
            if isinstance(agents[player_id], PPO):
                reward = [time_step.rewards[agents[player_id].player_id]]
                done = [time_step.last()]
                agents[player_id].post_step(reward, done)

                # Trigger learning when batch is full
                if agents[player_id].cur_batch_idx >= agents[player_id].steps_per_batch:
                    agents[player_id].learn([time_step])

        # Episode is over, step all agents with final info state.
        for agent in agents:
            if isinstance(agent, PPO):
                # Don't step PPO at terminal - it already processed the final step
                pass
            else:
                agent.step(time_step)


main()
