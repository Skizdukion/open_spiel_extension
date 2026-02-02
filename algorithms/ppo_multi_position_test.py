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
"""Tests for PPOMultiPosition."""
import os
import sys

# Add parent directory to path so we can import from algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from absl.testing import absltest
import numpy as np
import torch

from open_spiel.python import rl_environment
from game.tictactoe import TicTacToeGame
from ppo_multi_position import PPOMultiPosition
from ppo_multi_position import PPOAgent
from util.canical_obs import tictactoe_canonical_obs

SEED = 24261711


class PPOMultiPositionTest(absltest.TestCase):

    def test_single_player_mode(self):
        """Test PPOMultiPosition with single player (should work like regular PPO)."""
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0],  # Single player
            num_envs=1,
            steps_per_batch=16,
        )

        # Play some episodes
        for _ in range(10):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if agent.should_act([time_step]):
                    agent_output = agent.step([time_step])[0]
                    prev_time_step = time_step
                    time_step = env.step([agent_output.action])
                    reward = [time_step.rewards[player_id]]
                    done = [time_step.last()]
                    agent.post_step(prev_time_step, reward, done)
                    if agent.should_learn():
                        agent.learn([time_step])
                else:
                    # Random action for opponent
                    legal = time_step.observations["legal_actions"][player_id]
                    action = np.random.choice(legal)
                    time_step = env.step([action])

    def test_multi_player_mode(self):
        """Test PPOMultiPosition handling both players."""
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0, 1],  # Both players
            num_envs=1,
            steps_per_batch=16,
        )

        # Verify both players are handled
        self.assertEqual(agent.player_ids, [0, 1])

        # Play some episodes with shared network
        total_steps = {0: 0, 1: 0}
        for _ in range(20):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                self.assertTrue(agent.should_act([time_step]))

                agent_output = agent.step([time_step])[0]
                prev_time_step = time_step
                time_step = env.step([agent_output.action])

                reward = [time_step.rewards[player_id]]
                done = [time_step.last()]
                agent.post_step(prev_time_step, reward, done)
                total_steps[player_id] += 1

                if agent.should_learn():
                    agent.learn([time_step])

        # Both players should have taken steps
        self.assertGreater(total_steps[0], 0)
        self.assertGreater(total_steps[1], 0)

    def test_separate_buffers(self):
        """Test that experiences are stored in separate buffers per player."""
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0, 1],
            num_envs=1,
            steps_per_batch=128,
        )

        # Initially buffers are empty
        self.assertEqual(agent.cur_batch_idx[0], 0)
        self.assertEqual(agent.cur_batch_idx[1], 0)

        # Play one episode
        time_step = env.reset()
        p0_steps = 0
        p1_steps = 0
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agent.step([time_step])[0]
            prev_time_step = time_step
            time_step = env.step([agent_output.action])
            reward = [time_step.rewards[player_id]]
            done = [time_step.last()]
            agent.post_step(prev_time_step, reward, done)
            if player_id == 0:
                p0_steps += 1
            else:
                p1_steps += 1

        # Buffers should reflect separate counts
        self.assertEqual(agent.cur_batch_idx[0], p0_steps)
        self.assertEqual(agent.cur_batch_idx[1], p1_steps)

    def test_evaluation_mode(self):
        """Test that evaluation mode doesn't store experiences."""
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0, 1],
            num_envs=1,
            steps_per_batch=128,
        )

        # Play an episode in evaluation mode
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step([time_step], is_evaluation=True)[0]
            time_step = env.step([agent_output.action])

        # Buffers should remain empty
        self.assertEqual(agent.cur_batch_idx[0], 0)
        self.assertEqual(agent.cur_batch_idx[1], 0)

    def test_learning_convergence(self):
        """Test that shared network learns to play TicTacToe reasonably."""
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0, 1],
            num_envs=1,
            steps_per_batch=64,
        )

        # Train for some episodes
        for _ in range(200):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agent.step([time_step])[0]
                prev_time_step = time_step
                time_step = env.step([agent_output.action])
                reward = [time_step.rewards[player_id]]
                done = [time_step.last()]
                agent.post_step(prev_time_step, reward, done)
                if agent.should_learn():
                    agent.learn([time_step])

        # Evaluate: agent should make legal moves and complete games
        games_completed = 0
        for _ in range(10):
            time_step = env.reset()
            steps = 0
            while not time_step.last() and steps < 20:
                agent_output = agent.step([time_step], is_evaluation=True)[0]
                # Verify action is legal
                player_id = time_step.observations["current_player"]
                legal_actions = time_step.observations["legal_actions"][player_id]
                self.assertIn(agent_output.action, legal_actions)
                time_step = env.step([agent_output.action])
                steps += 1
            if time_step.last():
                games_completed += 1

        self.assertEqual(games_completed, 10)

    def test_canonical_form_pipeline(self):
        """Test that canonical form transformation works correctly.

        TicTacToe observations are 27-dim: [empty(9), P0_pieces(9), P1_pieces(9)]
        For canonical form, P1 should see channels swapped so their pieces are in channel 1.
        """
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0, 1],
            num_envs=1,
            steps_per_batch=32,
            canonical_obs_fn=tictactoe_canonical_obs,
        )

        # Play a full game and verify canonical transformation
        time_step = env.reset()
        move_count = 0
        canonical_verified = False

        while not time_step.last():
            player_id = time_step.observations["current_player"]
            raw_obs = time_step.observations["info_state"][player_id]

            # Verify canonical transformation for P1
            if player_id == 1 and move_count > 0:
                canonical_obs = agent._to_canonical_obs(raw_obs, player_id)
                # After canonical transform, P1's pieces should be in channel 1 (indices 9-17)
                # and P0's pieces should be in channel 2 (indices 18-26)
                self.assertTrue(np.array_equal(canonical_obs[9:18], raw_obs[18:27]))
                self.assertTrue(np.array_equal(canonical_obs[18:27], raw_obs[9:18]))
                canonical_verified = True

            agent_output = agent.step([time_step])[0]
            prev_time_step = time_step
            time_step = env.step([agent_output.action])

            reward = [time_step.rewards[player_id]]
            done = [time_step.last()]
            agent.post_step(prev_time_step, reward, done)
            move_count += 1

            if agent.should_learn():
                agent.learn([time_step])

        # Ensure canonical was verified at least once
        self.assertTrue(canonical_verified)

    def test_canonical_form_learning(self):
        """Test that agent can learn with canonical form over multiple games."""
        game = TicTacToeGame()
        env = rl_environment.Environment(game=game)

        def tictactoe_canonical_obs(obs, player_id):
            if player_id == 0:
                return obs
            obs = np.array(obs)
            channel_p0 = obs[9:18].copy()
            channel_p1 = obs[18:27].copy()
            obs[9:18] = channel_p1
            obs[18:27] = channel_p0
            return obs

        info_state_shape = (env.observation_spec()["info_state"][0],)
        num_actions = env.action_spec()["num_actions"]

        agent = PPOMultiPosition(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=2,
            player_ids=[0, 1],
            num_envs=1,
            steps_per_batch=64,
            canonical_obs_fn=tictactoe_canonical_obs,
        )

        # Train for multiple episodes
        learn_calls = {0: 0, 1: 0}
        for _ in range(50):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agent.step([time_step])[0]
                prev_time_step = time_step
                time_step = env.step([agent_output.action])
                reward = [time_step.rewards[player_id]]
                done = [time_step.last()]
                agent.post_step(prev_time_step, reward, done)

                # Track learning for each player
                for pid in [0, 1]:
                    if agent.cur_batch_idx[pid] >= agent.steps_per_batch:
                        learn_calls[pid] += 1

                if agent.should_learn():
                    agent.learn([time_step])

        # Both players should have learned
        self.assertGreater(learn_calls[0], 0)
        self.assertGreater(learn_calls[1], 0)


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    absltest.main()
