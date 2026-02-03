import os
import sys

# Add parent directory to path so we can import from algorithms
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tqdm import trange

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from algorithms import nfsp
from util.exploit_calculation import NSFPPoliciesEvaluate
from util.eval import EvalAgainstRandomBot, EvalAgainstOtherAgents
from util.setup_log import setup_log

logging = setup_log()

# ============== Configuration ==============
NUM_TRAIN_EPISODES = int(3e6)
EVAL_EVERY = 100000
HIDDEN_LAYERS_SIZES = [128]
REPLAY_BUFFER_CAPACITY = int(2e5)
RESERVOIR_BUFFER_CAPACITY = int(2e6)
ANTICIPATORY_PARAM = 0.05
CHECKPOINT_DIR = "checkpoints/nfsp_tictactoe"
SAVE_EVERY = 100000
LOAD_CHECKPOINT = None  # e.g., "checkpoints/nfsp_tictactoe/ep_100000"
# ===========================================


def save_checkpoint(agents, checkpoint_dir, episode):
    """Save all agents to checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ep_{episode}")
    for idx, agent in enumerate(agents):
        agent.save(f"{checkpoint_path}_agent{idx}")
    logging.info(f"Checkpoint saved at episode {episode}")


def load_checkpoint(agents, checkpoint_path):
    """Load all agents from checkpoint."""
    import pathlib

    for idx, agent in enumerate(agents):
        agent.restore(pathlib.Path(f"{checkpoint_path}_agent{idx}"))
    logging.info(f"Checkpoint loaded from {checkpoint_path}")


def main():
    game = "tic_tac_toe"
    num_players = 2

    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    kwargs = {
        "replay_buffer_capacity": REPLAY_BUFFER_CAPACITY,
        "epsilon_decay_duration": NUM_TRAIN_EPISODES,
        "epsilon_start": 0.5,
        "epsilon_end": 0.1,
    }

    agents = [
        nfsp.NFSP(
            idx,
            info_state_size,
            num_actions,
            HIDDEN_LAYERS_SIZES,
            RESERVOIR_BUFFER_CAPACITY,
            ANTICIPATORY_PARAM,
            **kwargs,
        )
        for idx in range(num_players)
    ]
    expl_policies_avg = NSFPPoliciesEvaluate(env, agents, nfsp.MODE.AVERAGE_POLICY)

    # Load checkpoint if specified
    if LOAD_CHECKPOINT:
        load_checkpoint(agents, LOAD_CHECKPOINT)

    random_bot_eval = EvalAgainstRandomBot(env, agents, 1000)
    others_agent_eval = EvalAgainstOtherAgents(env, agents, 1000)

    for ep in trange(NUM_TRAIN_EPISODES):
        if (ep + 1) % EVAL_EVERY == 0:
            losses = [agent.loss for agent in agents]
            logging.info("Losses: %s", losses)
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            logging.info("[%s] Exploitability AVG %s", ep + 1, expl)

            wins, loses, draws = random_bot_eval.run_eval()
            logging.info("Against random agent ----------------------")
            for idx, _ in enumerate(wins):
                logging.info(
                    f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}"
                )

            wins, loses, draws = others_agent_eval.run_eval(agents)
            logging.info("Against others agent ----------------------")
            for idx, _ in enumerate(wins):
                logging.info(
                    f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}"
                )
            logging.info("_____________________________________________")

        # Save checkpoint
        if (ep + 1) % SAVE_EVERY == 0:
            save_checkpoint(agents, CHECKPOINT_DIR, ep + 1)

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


if __name__ == "__main__":
    main()
