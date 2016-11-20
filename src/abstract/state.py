from abstract.agent import Agent


class State(object):
    @property
    def actions(self):
        raise NotImplementedError()

    def execute(self, agent: Agent, action) -> 'State':
        raise NotImplementedError()

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def generate() -> 'State':
        raise NotImplementedError()
