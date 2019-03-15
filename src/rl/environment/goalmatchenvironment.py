from abc import abstractmethod

from rl.environment.goalenvironment import GoalEnvironment


class GoalMatchEnvironment(GoalEnvironment):
    @property
    @abstractmethod
    def winner(self):
        raise NotImplementedError()
