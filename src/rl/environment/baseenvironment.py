from abc import abstractmethod, ABC


class BaseEnvironment(ABC):
    @property
    @abstractmethod
    def actions(self):
        raise NotImplementedError()

    @abstractmethod
    def execute(self, agent: 'BaseAgent', action) -> 'BaseEnvironment':
        raise NotImplementedError()
