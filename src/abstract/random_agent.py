from abstract.agent import Agent
from abstract.state import State
import random


class RandomAgent(Agent):
    def policy(self, state: State, **kwargs):
        return random.choice(state.actions)
