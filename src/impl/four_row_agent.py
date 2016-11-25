import random
from q_agent import QAgent
from impl.four_row_state import FourRowState
import numpy as np


class FourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions):
        raise NotImplementedError()

    def _select_learning_action(self, state: FourRowState):
        raise NotImplementedError()

    def _reward(self, state: FourRowState, action, new_state: FourRowState) -> float:
        d = float(new_state._discs_filled)
        reward = -d

        if new_state.is_terminal:
            if new_state.winner == self.identifier:
                reward = 100.0 * (1.0 / d)
            elif new_state.winner == -1:
                reward = -d
            else:
                reward = -100.0 * d

        return reward
        #return -10 * self._get_contiguous_enemy_entries(new_state) + 10 * self._get_contiguous_entries(new_state)


class EpsilonGreedyFourRowAgent(FourRowAgent):
    def _select_action_from_best(self, state: FourRowState, actions):
        return random.choice(actions)

    def _select_learning_action(self, state: FourRowState):
        epsilon = 0.3

        if random.random() < epsilon:
            return random.choice(state.actions)

        return max(state.actions, key=lambda action: self._q(state, action))


class SoftmaxFourRowAgent(FourRowAgent):
    def __init__(self, temperature, *args, **kwargs):
        super(FourRowAgent, self).__init__(*args, **kwargs)
        self._iteration_counter = 0
        self._temperature = temperature

    def _select_action_from_best(self, state: FourRowState, actions):
        return random.choice(actions)

    @staticmethod
    def _softmax(w, t=1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    def _select_learning_action(self, state: FourRowState):
        self._iteration_counter += 1
        actions = [self._q(state, action) for action in state.actions]
        pairs = zip(state.actions, self._softmax(actions, self._temperature(self, self._iteration_counter)))
        return max(pairs, key=lambda x: x[1])[0]
