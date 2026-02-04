import os
import sys
import argparse

# Add parent directory to path so we can import from algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import trange
from game.gomuko import GomukoGame
from util.eval import EvalAgainstRandomBot, EvalAgainstOtherAgents
from open_spiel.python import rl_environment, rl_agent
from util.setup_log import setup_log
from util.opponent_pool import HistoricalOpponentPool
from algorithms.dqn import DQN
import numpy as np
from open_spiel.python.algorithms import random_agent

logging = setup_log()


def parse_args():
    parser = argparse.ArgumentParser(description="Gomuko DQN Solver")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="gomuko/checkpoints",
        help="Directory to save model checkpoints (default: gomuko/checkpoints)",
    )
    return parser.parse_args()


def sample_agent(
    dqn_agents: list[rl_agent],
    opponent_pools: list[HistoricalOpponentPool],
    random_agents: list[random_agent.RandomAgent],
    self_play_prob=0.6,
    his_prob=0.35,
):
    roll = np.random.random()
    game_agents = [dqn_agents[0], dqn_agents[1]]

    if roll < self_play_prob:
        return game_agents

    training_player = np.random.randint(0, 2)
    opponent_player = 1 - training_player

    if roll < self_play_prob + his_prob and len(opponent_pools[opponent_player]) > 0:
        game_agents[opponent_player] = opponent_pools[opponent_player].sample_opponent()
    else:

        game_agents[opponent_player] = random_agents[opponent_player]

    return game_agents


def main(args):
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    tictactoe = GomukoGame()

    num_players = 2
    env = rl_environment.Environment(tictactoe)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [128, 128]
    replay_buffer_capacity = int(3e6)
    train_episodes = int(1e6)
    loss_report_interval = int(2e4)
    save_model_interval = int(2e5)
    eval_interval = int(1e5)
    snappot_opp_interval = int(1e5)

    dqn_agents = [
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
            optimizer_str="adam",
            # device="cuda",
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
            optimizer_str="adam",
            # device="cuda",
        ),
    ]

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    random_bot_eval = EvalAgainstRandomBot(env, dqn_agents, 1000)
    others_agent_eval = EvalAgainstOtherAgents(env, dqn_agents, 1000)

    opponent_pools = [HistoricalOpponentPool(max_size=10) for _ in range(num_players)]

    for ep in trange(train_episodes):
        if ep and ep % loss_report_interval == 0:
            q1 = dqn_agents[0].q_values
            q2 = dqn_agents[1].q_values
            q1_mean = q1.mean().item() if q1 is not None else 0
            q2_mean = q2.mean().item() if q2 is not None else 0
            logging.info(
                "[%s/%s] DQN 1 loss: %s, Q-mean: %.4f",
                ep,
                train_episodes,
                dqn_agents[0].loss,
                q1_mean,
            )
            logging.info(
                "[%s/%s] DQN 2 loss: %s, Q-mean: %.4f",
                ep,
                train_episodes,
                dqn_agents[1].loss,
                q2_mean,
            )

        if ep and ep % eval_interval == 0:
            logging.info("Against random agent ----------------------")
            wins, loses, draws = random_bot_eval.run_eval()
            for idx, _ in enumerate(wins):
                logging.info(
                    f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}"
                )

            wins, loses, draws = others_agent_eval.run_eval(dqn_agents)
            logging.info("Against others agent ----------------------")
            for idx, _ in enumerate(wins):
                logging.info(
                    f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}"
                )

        if ep and ep % save_model_interval == 0:
            dqn_agents[0].save(f"{checkpoint_dir}/dqn_agent_{0}_checkpoint_{ep}.pt")
            dqn_agents[1].save(f"{checkpoint_dir}/dqn_agent_{1}_checkpoint_{ep}.pt")
            logging.info(f"  Models saved at episode {ep} to {checkpoint_dir}")

        if ep > 0 and ep % snappot_opp_interval == 0:
            for player_id in range(num_players):
                opponent_pools[player_id].add_snapshot(dqn_agents[player_id])

        time_step = env.reset()

        cur_agents = sample_agent(dqn_agents, opponent_pools, random_agents)

        while not time_step.last():
            player_id = time_step.observations["current_player"]

            # Check if the current agent is one of the training agents
            is_training_agent = cur_agents[player_id] in dqn_agents

            agent_output = cur_agents[player_id].step(
                time_step, is_evaluation=not is_training_agent
            )

            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in dqn_agents:
            agent.step(time_step)


if __name__ == "__main__":
    args = parse_args()
    main(args)
