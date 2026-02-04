import os
import sys

# Add parent directory to path so we can import from algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from game.tictactoe import TicTacToeGame
from util.eval import EvalAgainstRandomBot, EvalAgainstOtherAgents
from util.exploit_calculation import DQNPoliciesEvaluate
from open_spiel.python import rl_environment
from util.setup_log import setup_log
from algorithms.dqn import DQN
from open_spiel.python.algorithms import exploitability

logging = setup_log()


def main():
    tictactoe = TicTacToeGame()

    env = rl_environment.Environment(tictactoe)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [32, 32]
    replay_buffer_capacity = int(1e4)
    train_episodes = 100000
    loss_report_interval = 10000

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
