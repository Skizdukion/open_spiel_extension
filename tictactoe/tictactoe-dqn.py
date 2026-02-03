from typing import List
import importlib.util
import os
import sys

# Add parent directory to path so we can import from algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from game.tictactoe import TicTacToeGame
from util.eval import EvalAgainstRandomBot, EvalAgainstOtherAgents
from util.exploit_calculation import DQNPoliciesEvaluate
import numpy as np
import random
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from util.setup_log import setup_log
from algorithms.dqn import DQN
from open_spiel.python.algorithms import exploitability

logging = setup_log()


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

    hidden_layers_sizes = [32, 32]
    replay_buffer_capacity = int(1e4)
    train_episodes = 100000
    loss_report_interval = 1000

    agents = [
        DQN(
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity,
        ),
        DQN(
            player_id=1,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity,
        ),
    ]

    for ep in trange(train_episodes):
        if ep and ep % loss_report_interval == 0:
            q1 = agents[0].q_values
            q2 = agents[1].q_values
            q1_mean = q1.mean().item() if q1 is not None else 0
            q2_mean = q2.mean().item() if q2 is not None else 0
            logging.info(
                "[%s/%s] DQN 1 loss: %s, Q-mean: %.4f",
                ep,
                train_episodes,
                agents[0].loss,
                q1_mean,
            )
            logging.info(
                "[%s/%s] DQN 2 loss: %s, Q-mean: %.4f",
                ep,
                train_episodes,
                agents[1].loss,
                q2_mean,
            )

        time_step = env.reset()

        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    random_bot_eval = EvalAgainstRandomBot(env, agents, 1000)

    wins, loses, draws = random_bot_eval.run_eval()

    logging.info("Against random agent ----------------------")
    for idx, _ in enumerate(wins):
        logging.info(
            f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}"
        )

    others_agent_eval = EvalAgainstOtherAgents(env, agents, 1000)

    wins, loses, draws = others_agent_eval.run_eval(agents)

    logging.info("Against others agent ----------------------")
    for idx, _ in enumerate(wins):
        logging.info(
            f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}"
        )

    policies = DQNPoliciesEvaluate(env, agents)

    expl = exploitability.exploitability(env.game, policies)
    logging.info("Exploitability AVG %s", expl)


main()
