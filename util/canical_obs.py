import numpy as np


def tictactoe_canonical_obs(obs, player_id):
    if player_id == 0:
        return obs
    # For P1, swap P0 channel (9-17) with P1 channel (18-26)
    obs = np.array(obs)
    channel_p0 = obs[9:18].copy()
    channel_p1 = obs[18:27].copy()
    obs[9:18] = channel_p1  # P1's pieces now in "my pieces" slot
    obs[18:27] = channel_p0  # P0's pieces now in "opponent pieces" slot
    return obs
