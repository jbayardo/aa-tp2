from abstract.agent import Agent
import random


class RandomAgent(Agent):
    def name(self) -> str:
        return 'Random Agent'

    def policy(self, state):
        return random.choice(state.actions)
