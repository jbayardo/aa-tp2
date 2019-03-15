import uuid
from abc import ABC, abstractmethod

from rl.environment.baseenvironment import BaseEnvironment


class BaseAgent(ABC):
    def __init__(self, identifier: int, name: str = None):
        assert identifier >= 0

        self._identifier = identifier

        self._name = name
        if self._name is None:
            self._name = str(uuid.uuid4())

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> int:
        return self._identifier

    @abstractmethod
    def policy(self, environment: BaseEnvironment, **kwargs):
        raise NotImplementedError()

    def feedback(self, previous: BaseEnvironment, action, current: BaseEnvironment, **kwargs) -> None:
        pass

    def epoch_statistics(self):
        return None

    def episode_statistics(self):
        return None

    def deploy(self) -> 'BaseAgent':
        return self
