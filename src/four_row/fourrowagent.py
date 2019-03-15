import random

import numpy as np

from rl.agent.qagent import DoubleQAgent
from four_row.fourrowenvironment import FourRowEnvironment


class FourRowAgent(DoubleQAgent):
    def _select_learning_action(self, environment: FourRowEnvironment, **kwargs):
        return environment.actions[0]

    def _reward(self, previous: FourRowEnvironment, action, current: FourRowEnvironment) -> float:
        d = float(current._discs_filled)
        reward = -d

        if current.is_terminal:
            if current.winner == self.identifier:
                reward = 1000.0 * (1.0 / d)
            elif current.winner == -1:
                reward = -d
            else:
                reward = -100.0 * d

        return reward
        #return -10 * self._get_contiguous_enemy_entries(new_state) + 10 * self._get_contiguous_entries(new_state)

    def _get_contiguous_entries(self, state: FourRowEnvironment, enemy: bool = False):
        pass


class EpsilonGreedyFourRowAgent(FourRowAgent):
    def __init__(self, epsilon: float, *args, **kwargs):
        assert 0 <= epsilon <= 1.0

        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    def _select_learning_action(self, environment: FourRowEnvironment, **kwargs):
        if random.random() < self._epsilon:
            return random.choice(environment.actions)

        return max(environment.actions, key=lambda action: self._q(environment, action))


class SoftmaxFourRowAgent(FourRowAgent):
    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    @staticmethod
    def _softmax(w, t = 1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    def _select_learning_action(self, environment: FourRowEnvironment, **kwargs):
        actions = [self._q(environment, action) for action in environment.actions]
        pairs = zip(environment.actions, self._softmax(actions, self._temperature(self, kwargs['episode'], kwargs['turns'])))
        return max(pairs, key=lambda x: x[1])[0]
