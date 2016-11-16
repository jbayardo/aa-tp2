from abstract.agent import Agent


class RandomAgent(Agent):
    def name(self) -> str:
        return 'Random Agent'

    def policy(self, state):
        import random
        return random.choice(state.actions)
