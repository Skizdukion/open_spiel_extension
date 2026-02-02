# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-position PPO implementation.

Supports training a single network to play as multiple player positions.
The agent automatically detects current player from time_step and uses
the appropriate perspective for observation extraction.
"""

import time
from typing import List

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

INVALID_ACTION_PENALTY = -1e6


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    """A masked categorical."""

    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PPOAgent(nn.Module):
    """A PPO agent module."""

    def __init__(self, num_actions, observation_shape, device):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )
        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        logits = self.actor(x)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(x),
            probs.probs,
        )


def legal_actions_to_mask(legal_actions_list, num_actions):
    """Converts a list of legal actions to a mask."""
    legal_actions_mask = torch.zeros(
        (len(legal_actions_list), num_actions), dtype=torch.bool
    )
    for i, legal_actions in enumerate(legal_actions_list):
        legal_actions_mask[i, legal_actions] = 1
    return legal_actions_mask


class PPOMultiPosition(nn.Module):
    """PPO Agent that can play as multiple player positions.

    Uses a single shared network for all positions. Automatically detects
    current player from time_step and uses correct observation perspective.

    When use_canonical_form=True, observations are transformed so all players
    see the game from the same perspective ("my pieces" vs "opponent pieces").
    This is essential for shared networks to learn effectively.

    Args:
        player_ids: List of player IDs this agent handles, e.g., [0, 1] for both players
        canonical_obs_fn: Optional function(obs, player_id) -> canonical_obs to transform
                         observations to canonical form. If None, uses raw observations.
    """

    def __init__(
        self,
        input_shape,
        num_actions,
        num_players,
        player_ids: List[int],  # e.g., [0, 1] for both players
        num_envs=1,
        steps_per_batch=128,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        clip_coef=0.2,
        clip_vloss=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        device="cpu",
        writer=None,
        agent_fn=PPOAgent,
        canonical_obs_fn=None,  # Function(obs, player_id) -> canonical_obs
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_players = num_players
        self.player_ids = list(player_ids)
        self.device = device
        self.canonical_obs_fn = canonical_obs_fn

        # Training settings
        self.num_envs = num_envs
        self.steps_per_batch = steps_per_batch
        self.batch_size = self.num_envs * self.steps_per_batch
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate

        # Loss function
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Logging
        self.writer = writer

        # Shared network for all positions
        self.network = agent_fn(self.num_actions, self.input_shape, device).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

        # Initialize training buffers - SEPARATE for each player position
        self._init_buffers()

        # Counters per position
        self.cur_batch_idx = {pid: 0 for pid in self.player_ids}
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()

    def _init_buffers(self):
        """Initialize separate experience buffers for each player position."""
        self.legal_actions_mask = {
            pid: torch.zeros(
                (self.steps_per_batch, self.num_envs, self.num_actions),
                dtype=torch.bool,
            ).to(self.device)
            for pid in self.player_ids
        }
        self.obs = {
            pid: torch.zeros(
                (self.steps_per_batch, self.num_envs) + self.input_shape
            ).to(self.device)
            for pid in self.player_ids
        }
        self.actions = {
            pid: torch.zeros((self.steps_per_batch, self.num_envs)).to(self.device)
            for pid in self.player_ids
        }
        self.logprobs = {
            pid: torch.zeros((self.steps_per_batch, self.num_envs)).to(self.device)
            for pid in self.player_ids
        }
        self.rewards = {
            pid: torch.zeros((self.steps_per_batch, self.num_envs)).to(self.device)
            for pid in self.player_ids
        }
        self.dones = {
            pid: torch.zeros((self.steps_per_batch, self.num_envs)).to(self.device)
            for pid in self.player_ids
        }
        self.values = {
            pid: torch.zeros((self.steps_per_batch, self.num_envs)).to(self.device)
            for pid in self.player_ids
        }

    def _get_current_player(self, time_step) -> int:
        """Extract current player from time_step."""
        if isinstance(time_step, list):
            return time_step[0].observations["current_player"]
        return time_step.observations["current_player"]

    def should_act(self, time_step) -> bool:
        """Check if this agent should act for the current player."""
        current_player = self._get_current_player(time_step)
        return current_player in self.player_ids

    def get_value(self, x):
        return self.network.get_value(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        return self.network.get_action_and_value(x, legal_actions_mask, action)

    def _to_canonical_obs(self, obs: np.ndarray, player_id: int) -> np.ndarray:
        """Transform observation to canonical form using the provided function.

        If canonical_obs_fn was provided at init, applies it to transform the
        observation. Otherwise returns the observation unchanged.

        Args:
            obs: Raw observation array
            player_id: Current player (0 or 1)

        Returns:
            Canonical observation (same shape)
        """
        if self.canonical_obs_fn is None:
            return obs
        return self.canonical_obs_fn(obs, player_id)

    def step(self, time_step, is_evaluation=False):
        """Take action, automatically using current player's perspective."""
        # Auto-detect current player
        player_id = self._get_current_player(time_step)

        if player_id not in self.player_ids:
            raise ValueError(
                f"Current player {player_id} not in player_ids {self.player_ids}"
            )

        if is_evaluation:
            with torch.no_grad():
                legal_actions_mask = legal_actions_to_mask(
                    [ts.observations["legal_actions"][player_id] for ts in time_step],
                    self.num_actions,
                ).to(self.device)
                obs = torch.Tensor(
                    np.array(
                        [
                            np.reshape(
                                self._to_canonical_obs(
                                    ts.observations["info_state"][player_id], player_id
                                ),
                                self.input_shape,
                            )
                            for ts in time_step
                        ]
                    )
                ).to(self.device)
                action, _, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )
                return [
                    StepOutput(action=a.item(), probs=p)
                    for (a, p) in zip(action, probs)
                ]
        else:
            with torch.no_grad():
                obs = torch.Tensor(
                    np.array(
                        [
                            np.reshape(
                                self._to_canonical_obs(
                                    ts.observations["info_state"][player_id], player_id
                                ),
                                self.input_shape,
                            )
                            for ts in time_step
                        ]
                    )
                ).to(self.device)
                legal_actions_mask = legal_actions_to_mask(
                    [ts.observations["legal_actions"][player_id] for ts in time_step],
                    self.num_actions,
                ).to(self.device)
                action, logprob, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )

                # Store in this player's buffers
                idx = self.cur_batch_idx[player_id]
                self.legal_actions_mask[player_id][idx] = legal_actions_mask
                self.obs[player_id][idx] = obs
                self.actions[player_id][idx] = action
                self.logprobs[player_id][idx] = logprob
                self.values[player_id][idx] = value.flatten()

                return [
                    StepOutput(action=a.item(), probs=p)
                    for (a, p) in zip(action, probs)
                ]

    def post_step(self, time_step, reward, done):
        """Store reward/done for current player. Call AFTER step()."""
        player_id = self._get_current_player(time_step)

        if player_id not in self.player_ids:
            return

        idx = self.cur_batch_idx[player_id]
        self.rewards[player_id][idx] = torch.tensor(reward).to(self.device).view(-1)
        self.dones[player_id][idx] = torch.tensor(done).to(self.device).view(-1)

        self.total_steps_done += self.num_envs
        self.cur_batch_idx[player_id] += 1

    def should_learn(self, player_id: int = None) -> bool:
        """Check if any player (or specific player) has full batch."""
        if player_id is not None:
            return self.cur_batch_idx[player_id] >= self.steps_per_batch
        return any(
            self.cur_batch_idx[pid] >= self.steps_per_batch for pid in self.player_ids
        )

    def learn(self, time_step=None):
        """Learn from all player positions that have full batches."""
        for player_id in self.player_ids:
            if self.cur_batch_idx[player_id] >= self.steps_per_batch:
                self._learn_for_player(player_id, time_step)

    def _learn_for_player(self, player_id: int, time_step=None):
        """Perform PPO update for a specific player's experiences."""
        # Get next observation for bootstrap
        if time_step is not None:
            next_obs = torch.Tensor(
                np.array(
                    [
                        np.reshape(
                            self._to_canonical_obs(
                                ts.observations["info_state"][player_id], player_id
                            ),
                            self.input_shape,
                        )
                        for ts in time_step
                    ]
                )
            ).to(self.device)
        else:
            # Use last observation as next_obs (terminal state approximation)
            next_obs = self.obs[player_id][self.steps_per_batch - 1]

        # Get buffers for this player
        obs = self.obs[player_id]
        actions = self.actions[player_id]
        logprobs = self.logprobs[player_id]
        rewards = self.rewards[player_id]
        dones = self.dones[player_id]
        values = self.values[player_id]
        legal_actions_masks = self.legal_actions_mask[player_id]

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_batch)):
                    nextvalues = (
                        next_value if t == self.steps_per_batch - 1 else values[t + 1]
                    )
                    nextnonterminal = 1.0 - dones[t]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(self.device)
                for t in reversed(range(self.steps_per_batch)):
                    next_return = (
                        next_value if t == self.steps_per_batch - 1 else returns[t + 1]
                    )
                    nextnonterminal = 1.0 - dones[t]
                    returns[t] = rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_legal_actions_mask = legal_actions_masks.reshape((-1, self.num_actions))
        b_obs = obs.reshape((-1,) + self.input_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
                    b_obs[mb_inds],
                    legal_actions_mask=b_legal_actions_mask[mb_inds],
                    action=b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.entropy_coef * entropy_loss
                    + v_loss * self.value_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        # Logging
        if self.writer is not None:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            self.writer.add_scalar(
                f"losses/value_loss_p{player_id}", v_loss.item(), self.total_steps_done
            )
            self.writer.add_scalar(
                f"losses/policy_loss_p{player_id}",
                pg_loss.item(),
                self.total_steps_done,
            )
            self.writer.add_scalar(
                f"losses/entropy_p{player_id}",
                entropy_loss.item(),
                self.total_steps_done,
            )
            self.writer.add_scalar(
                f"losses/explained_variance_p{player_id}",
                explained_var,
                self.total_steps_done,
            )

        # Reset buffer index for this player
        self.updates_done += 1
        self.cur_batch_idx[player_id] = 0

    def anneal_learning_rate(self, update, num_total_updates):
        frac = 1.0 - (update / num_total_updates)
        if frac <= 0:
            raise ValueError("Annealing learning rate to <= 0")
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow
