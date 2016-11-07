from typing import List
from q_agent import QAgent
from impl.four_row_action import FourRowAction
from impl.four_row_state import FourRowState
import random


class FourRowAgent(QAgent):
    def _select_action_from_best(self, state: FourRowState, actions: List[FourRowAction]) -> FourRowAction:
        return actions[0]

    def _select_learning_action(self, state: FourRowState) -> FourRowAction:
        return random.choice(state.actions)

    def _reward(self, state: FourRowState, action: FourRowAction, new_state: FourRowState) -> float:
        if new_state.is_terminal:
            return 1.0
        return 0.0
