from abstract.action import Action
from abstract.agent import Agent
from abstract.state import State


class RandomAgent(Agent):
    def name(self) -> str:
        return 'Random Agent'

    def policy(self, state: State) -> Action:
        import random
        return random.choice(state.actions)
