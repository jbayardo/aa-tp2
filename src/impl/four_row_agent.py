import random
from q_agent import QAgent
from impl.four_row_action import FourRowAction
from impl.four_row_state import FourRowState


class FourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions: FourRowAction) -> FourRowAction:
        raise NotImplementedError()

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        raise NotImplementedError()

    def _reward(self, state: FourRowState, action: FourRowAction, new_state: FourRowState) -> float:

        if new_state.is_terminal:
            if new_state.winner == self.identifier:
                return 100 * (1.0 / float(new_state._discs_filled))
            elif new_state.winner == -1:
                return -10
            else:
                return -100 * new_state._discs_filled

        return 0.0
        #return -10 * self._get_contiguous_enemy_entries(new_state) + 10 * self._get_contiguous_entries(new_state)


class EpsilonGreedyFourRowAgent(FourRowAgent):
    def _select_action_from_best(self, state: FourRowState, actions: FourRowAction) -> FourRowAction:
        return random.choice(actions)

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        epsilon = 0.3

        if random.random() < epsilon:
            return random.choice(state.actions)

        return max(state.actions, key=lambda action: self._q(state, action))


class SoftmaxFourRowAgent(FourRowAgent):
    def __init__(self, *args, **kwargs):
        super(FourRowAgent, self).__init__(*args, **kwargs)
        self._iteration_counter = 0
        self._temperature = 1.0

    def _select_action_from_best(self, state: FourRowState, actions: FourRowAction) -> FourRowAction:
        return random.choice(actions)

    @staticmethod
    def _softmax(w, t=1.0):
        import numpy as np
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        self._iteration_counter += 1
        actions = [self._q(state, action) for action in state.actions]
        pairs = zip(state.actions, self._softmax(actions, self._temperature))
        return max(pairs, key=lambda x: x[1])[0]
