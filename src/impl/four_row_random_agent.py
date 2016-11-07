from abstract.agent import Agent
from impl.four_row_action import FourRowAction
from impl.four_row_state import FourRowState
import random


class FourRowRandomAgent(Agent):
    def name(self) -> str:
        return 'Random Agent'

    def policy(self, state: FourRowState) -> FourRowAction:
        return random.choice(state.actions)
