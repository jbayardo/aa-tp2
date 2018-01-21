import random

from rl.agent import Agent
from rl.environment import Environment


class RandomAgent(Agent):
    def policy(self, environment: Environment, **kwargs):
        return random.choice(environment.actions)
