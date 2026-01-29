from open_spiel.python.games.tic_tac_toe import TicTacToeGame, TicTacToeState
import numpy as np
import random

tictactoe = TicTacToeGame()

state = tictactoe.new_initial_state()

# Print the initial state
print(str(state))

while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        num_actions = len(outcomes)
        print("Chance node, got " + str(num_actions) + " outcomes")
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        print(
            "Sampled outcome: ", state.action_to_string(state.current_player(), action)
        )
        state.apply_action(action)

    elif state.is_simultaneous_node():
        # Simultaneous node: sample actions for all players.

        def random_choice(a):
            return np.random.choice(a) if a else [0]

        chosen_actions = [
            random_choice(state.legal_actions(pid))
            for pid in range(tictactoe.num_players())
        ]
        print(
            "Chosen actions: ",
            [
                state.action_to_string(pid, action)
                for pid, action in enumerate(chosen_actions)
            ],
        )
        state.apply_actions(chosen_actions)
    else:
        # Decision node: sample action for the single current player
        action = random.choice(state.legal_actions(state.current_player()))
        action_string = state.action_to_string(state.current_player(), action)
        print(
            "Player ",
            state.current_player(),
            ", randomly sampled action: ",
            action_string,
        )
        state.apply_action(action)

    print(str(state))

# Game is now done. Print utilities for each player
returns = state.returns()
for pid in range(tictactoe.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))
