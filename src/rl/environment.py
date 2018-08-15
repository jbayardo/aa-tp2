class Environment(object):
    def actions(self, agent: 'Agent'):
        raise NotImplementedError()

    def execute(self, agent: 'Agent', action) -> 'Environment':
        raise NotImplementedError()

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def generate() -> 'Environment':
        raise NotImplementedError()
