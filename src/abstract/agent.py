from abstract.state import State
import uuid


class Agent(object):
    def __init__(self, identifier: int):
        self._identifier = identifier
        self._name = str(uuid.uuid4())

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> int:
        return self._identifier

    def policy(self, state: State):
        raise NotImplementedError()
