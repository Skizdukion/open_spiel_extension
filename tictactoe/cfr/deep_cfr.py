import os
import sys

# Add parent directory to path so we can import from algorithms
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from util.setup_log import setup_log
from algorithms import deep_cfr
from open_spiel.python import policy, rl_environment
import pyspiel
from open_spiel.python.algorithms import expected_game_score
from util.eval import EvalAgainstRandomBot, EvalAgainstOtherAgents

logging = setup_log()

# ============== Configuration ==============
CHECKPOINT_DIR = "checkpoints/deep_cfr_tictactoe"
LOAD_CHECKPOINT = None  # e.g., "checkpoints/deep_cfr_tictactoe/model.pt"
# ===========================================


class DeepCFRAgent:
    """Wrapper to make DeepCFRSolver compatible with eval utilities."""

    def __init__(self, solver, player_id, num_actions):
        self._solver = solver
        self.player_id = player_id
        self._num_actions = num_actions

    def step(self, time_step, is_evaluation=False):
        """Return action based on the policy network."""
        from open_spiel.python import rl_agent
        import numpy as np

        if time_step.last():
            return rl_agent.StepOutput(action=0, probs=np.zeros(self._num_actions))

        # Create a mock state from time_step observations
        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]

        # Get action probabilities from policy network
        with torch.no_grad():
            info_state_tensor = torch.FloatTensor(info_state).unsqueeze(0)
            logits = self._solver._policy_network(info_state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        # Mask illegal actions
        legal_probs = np.zeros(self._num_actions)
        for a in legal_actions:
            legal_probs[a] = probs[a]
        legal_probs /= legal_probs.sum()

        action = np.random.choice(self._num_actions, p=legal_probs)
        return rl_agent.StepOutput(action=action, probs=legal_probs)


def main():
    game = pyspiel.load_game("tic_tac_toe")
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(64,),
        advantage_network_layers=(64,),
        num_iterations=50,
        num_traversals=500,
        reinitialize_advantage_networks=True,
        learning_rate=1e-3,
        batch_size_advantage=256,
        batch_size_strategy=256,
        memory_capacity=100000,
        policy_network_train_steps=5000,
        advantage_network_train_steps=750,
    )

    # Load checkpoint if specified
    if LOAD_CHECKPOINT and os.path.exists(LOAD_CHECKPOINT):
        deep_cfr_solver.load(LOAD_CHECKPOINT)
        logging.info(f"Loaded checkpoint from {LOAD_CHECKPOINT}")
    else:
        # Train from scratch
        _, advantage_losses, policy_loss = deep_cfr_solver.solve()

        for player, losses in advantage_losses.items():
            logging.info(
                "Advantage for player %d: %s",
                player,
                losses[:2] + ["..."] + losses[-2:],
            )
            assert deep_cfr_solver.advantage_buffers[player] is not None
            logging.info(
                f"Advantage Buffer Size for player {player}:"
                f" {len(deep_cfr_solver.advantage_buffers[player])}"
            )

        logging.info(f"Strategy Buffer Size: {len(deep_cfr_solver.strategy_buffer)}")
        logging.info(f"Final policy loss: {policy_loss}")

        # Save the model
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
        deep_cfr_solver.save(checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")

    # Compute NashConv
    average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities
    )
    pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
    conv = pyspiel.nash_conv(game, pyspiel_policy)
    logging.info(f"Deep CFR in Tictactoe - NashConv: {conv}")

    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2
    )
    logging.info(f"Computed player 0 value: {average_policy_values[0]:.2f}")
    logging.info(f"Computed player 1 value: {average_policy_values[1]:.2f}")

    # Evaluation against random bot and self-play
    agents = [DeepCFRAgent(deep_cfr_solver, idx, num_actions) for idx in range(2)]

    random_bot_eval = EvalAgainstRandomBot(env, agents, 1000)
    wins, loses, draws = random_bot_eval.run_eval()
    logging.info("Against random agent ----------------------")
    for idx, _ in enumerate(wins):
        logging.info(f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}")

    others_agent_eval = EvalAgainstOtherAgents(env, agents, 1000)
    wins, loses, draws = others_agent_eval.run_eval(agents)
    logging.info("Against self (Deep CFR vs Deep CFR) ----------------------")
    for idx, _ in enumerate(wins):
        logging.info(f"P{idx} Win: {wins[idx]}, Lose: {loses[idx]}, Draw: {draws[idx]}")
    logging.info("_____________________________________________")


if __name__ == "__main__":
    main()
