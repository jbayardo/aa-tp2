from abstract.action import Action


class Agent(object):
    def __init__(self, identifier):
        self._identifier = identifier

    def name(self) -> str:
        raise NotImplementedError()

    @property
    def identifier(self):
        return self._identifier

    def policy(self, state) -> Action:
        raise NotImplementedError()
