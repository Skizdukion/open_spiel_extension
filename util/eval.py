from open_spiel.python import rl_environment, rl_agent
from open_spiel.python.algorithms import random_agent
import numpy as np


def run_simulation(agents, player_pos, num_eps, env):
    win = 0
    lose = 0

    for _ in range(num_eps):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])

        if time_step.rewards[player_pos] > 0:
            win += 1

        if time_step.rewards[player_pos] < 0:
            lose += 1

    return win, lose, num_eps - win - lose


class EvalAgainstRandomBot:
    def __init__(
        self,
        env: rl_environment.Environment,
        agents: list[rl_agent.AbstractAgent],
        num_episodes: int,
    ):
        self._env = env
        self._agents = agents
        self._num_player = len(agents)
        self.num_episodes = num_episodes
        num_actions = env.action_spec()["num_actions"]
        self._random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(self._num_player)
        ]

    def run_eval(self):
        wins = np.zeros(self._num_player)
        loses = np.zeros(self._num_player)
        draws = np.zeros(self._num_player)

        for player_pos in range(self._num_player):
            cur_agents = self.create_cur_agents(player_pos)

            win, lose, draw = run_simulation(
                cur_agents, player_pos, self.num_episodes, self._env
            )
            wins[player_pos] = win
            loses[player_pos] = lose
            draws[player_pos] = draw

        return wins, loses, draws

    def create_cur_agents(self, select_agents_to_eval: int):
        agents = []
        for player_pos in range(self._num_player):
            if player_pos == select_agents_to_eval:
                agents.append(self._agents[player_pos])
            else:
                agents.append(self._random_agents[player_pos])

        return agents


class EvalAgainstOtherAgents:
    def __init__(
        self,
        env: rl_environment.Environment,
        agents: list[rl_agent.AbstractAgent],
        num_episodes: int,
    ):
        self._env = env
        self._agents = agents
        self.num_episodes = num_episodes
        self.num_actions = env.action_spec()["num_actions"]
        self._num_player = len(agents)

    def run_eval(self, other_agents: list[rl_agent.AbstractAgent]):
        assert len(other_agents) == self._num_player

        wins = np.zeros(self._num_player)
        loses = np.zeros(self._num_player)
        draws = np.zeros(self._num_player)

        for player_pos in range(self._num_player):
            cur_agents = self.create_cur_agents(player_pos, other_agents)

            win, lose, draw = run_simulation(
                cur_agents, player_pos, self.num_episodes, self._env
            )
            wins[player_pos] = win
            loses[player_pos] = lose
            draws[player_pos] = draw

        return wins, loses, draws

    def create_cur_agents(
        self, select_agents_to_eval: int, other_agents: list[rl_agent.AbstractAgent]
    ):
        agents = []
        for player_pos in range(self._num_player):
            if player_pos == select_agents_to_eval:
                agents.append(self._agents[player_pos])
            else:
                agents.append(other_agents[player_pos])

        return agents
