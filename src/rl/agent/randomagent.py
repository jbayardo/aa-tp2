import random

from rl.agent.baseagent import BaseAgent
from rl.environment.baseenvironment import BaseEnvironment


class RandomAgent(BaseAgent):
    def policy(self, environment: BaseEnvironment, **kwargs):
        return random.choice(environment.actions)
