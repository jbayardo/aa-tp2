from typing import List
import random
from q_agent import QAgent
from impl.four_row_action import FourRowAction
from impl.four_row_state import FourRowState


class EpsilonGreedyFourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions: List[FourRowAction]) -> FourRowAction:
        return random.choice(actions)

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        epsilon = 0.3

        if random.random() < epsilon:
            return random.choice(state.actions)

        return max(state.actions, key=lambda action: self._q(state, action))

    def _reward(self, state: FourRowState, action: FourRowAction, new_state: FourRowState) -> float:
        if new_state.is_terminal:
            return new_state._discs_filled
        return 0.0


class EpsilonFirstFourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions: List[FourRowAction]) -> FourRowAction:
        return random.choice(actions)

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        epsilon = 0.3

        if random.random() < epsilon:
            return random.choice(state.actions)

        return max(state.actions, key=lambda action: self._q(state, action))

    def _reward(self, state: FourRowState, action: FourRowAction, new_state: FourRowState) -> float:
        if new_state.is_terminal:
            return new_state._discs_filled
        return 0.0


class SoftmaxFourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions: List[FourRowAction]) -> FourRowAction:
        return random.choice(actions)

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        epsilon = 0.3

        if random.random() < epsilon:
            return random.choice(state.actions)

        return max(state.actions, key=lambda action: self._q(state, action))

    def _reward(self, state: FourRowState, action: FourRowAction, new_state: FourRowState) -> float:
        if new_state.is_terminal:
            return new_state._discs_filled
        return 0.0
