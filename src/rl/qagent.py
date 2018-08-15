import logging
import os

import numpy as np

from rl.agent import Agent
from rl.environment import Environment
from rl.qtable import QTable
from rl.fixedevaluationqagent import FixedEvaluationQAgent


class QAgent(Agent):
    def __init__(self, q: QTable = None, *args, **kwargs):
        assert 'learning_rate' in kwargs
        assert 'discount_factor' in kwargs

        super(QAgent, self).__init__(*args, **kwargs)

        if q is None:
            q = QTable()
        self._q = q

        self._learning_rate = kwargs['learning_rate']
        self._discount_factor = kwargs['discount_factor']

    def policy(self, environment: Environment, **kwargs):
        assert not environment.is_terminal
        return self._select_learning_action(environment, **kwargs)

    def feedback(self, previous: Environment, action, current: Environment, **kwargs) -> None:
        episode = kwargs['episode']
        turns = kwargs['turns']

        reward = self._reward(previous, action, current)
        scores = np.array([self._q(current, action) for action in current.actions(self)])

        maximum_factor = 0.0
        if len(scores) > 0:
            maximum_factor = np.amax(scores)

        current_q = self._q(previous, action)

        learning_rate = self._learning_rate(self, episode, turns)
        discount_factor = self._discount_factor(self, episode, turns)

        # noinspection PyTypeChecker
        self._q.update(previous, action,
                       current_q + learning_rate * (reward + discount_factor * maximum_factor - current_q))

    def _q_average(self):
        return self._q.average()

    def _reward(self, state: Environment, action, new_state: Environment) -> np.float64:
        raise NotImplementedError()

    def _select_learning_action(self, state: Environment, **kwargs):
        raise NotImplementedError()

    def deploy(self) -> FixedEvaluationQAgent:
        return FixedEvaluationQAgent(self._q, identifier=self.identifier, name=self.name)
