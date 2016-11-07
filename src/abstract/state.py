from typing import Iterable
from abstract.action import Action
from abstract.agent import Agent


class State(object):
    @property
    def actions(self) -> Iterable[Action]:
        raise NotImplementedError()

    def execute(self, agent: Agent, action: Action) -> 'State':
        raise NotImplementedError()

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def generate() -> 'State':
        raise NotImplementedError()
