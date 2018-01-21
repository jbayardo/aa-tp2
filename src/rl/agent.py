import logging
import uuid

from rl.environment import Environment


class Agent(object):
    def __init__(self, *args, **kwargs):
        assert 'identifier' in kwargs
        assert isinstance(kwargs['identifier'], int)
        assert kwargs['identifier'] >= 0

        self._identifier = kwargs['identifier']

        if 'name' in kwargs:
            self._name = kwargs['name']
        else:
            self._name = uuid.uuid4()
        self._name = str(self._name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> int:
        return self._identifier

    def policy(self, environment: Environment, **kwargs):
        raise NotImplementedError()

    def feedback(self, previous: Environment, action, current: Environment, **kwargs) -> None:
        pass

    def deploy(self) -> 'Agent':
        return self
