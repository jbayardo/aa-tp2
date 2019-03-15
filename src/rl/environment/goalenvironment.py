from abc import abstractmethod

from rl.environment.baseenvironment import BaseEnvironment


class GoalEnvironment(BaseEnvironment):
    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError()
