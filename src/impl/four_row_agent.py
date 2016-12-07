import random

import numpy as np

from abstract.q_agent import QAgent
from impl.four_row_state import FourRowState


class FourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions):
        return actions[0]

    def _select_learning_action(self, state: FourRowState, **kwargs):
        return state.actions[0]

    def _reward(self, state: FourRowState, action, new_state: FourRowState) -> float:
        d = float(new_state._discs_filled)
        reward = -d

        if new_state.is_terminal:
            if new_state.winner == self.identifier:
                reward = 1000.0 * (1.0 / d)
            elif new_state.winner == -1:
                reward = -d
            else:
                reward = -100.0 * d

        return reward
        #return -10 * self._get_contiguous_enemy_entries(new_state) + 10 * self._get_contiguous_entries(new_state)

    def _get_contiguous_entries(self, state: FourRowState, enemy: bool = False):
        pass


class EpsilonGreedyFourRowAgent(FourRowAgent):
    def __init__(self, *args, **kwargs):
        super(EpsilonGreedyFourRowAgent, self).__init__(*args, **kwargs)
        self._epsilon = kwargs['epsilon']

        assert self._epsilon >= 0.0
        assert self._epsilon <= 1.0

    def _select_action_from_best(self, state: FourRowState, actions):
        return random.choice(actions)

    def _select_learning_action(self, state: FourRowState, **kwargs):
        if random.random() < self._epsilon:
            return random.choice(state.actions)

        return max(state.actions, key=lambda action: self._q(state, action))


class SoftmaxFourRowAgent(FourRowAgent):
    def __init__(self, *args, **kwargs):
        super(SoftmaxFourRowAgent, self).__init__(*args, **kwargs)
        self._temperature = kwargs['temperature']

    def _select_action_from_best(self, state: FourRowState, actions):
        return random.choice(actions)

    @staticmethod
    def _softmax(w, t = 1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    def _select_learning_action(self, state: FourRowState, **kwargs):
        actions = [self._q(state, action) for action in state.actions]
        pairs = zip(state.actions, self._softmax(actions, self._temperature(self, kwargs['episode'], kwargs['turns'])))
        return max(pairs, key=lambda x: x[1])[0]
