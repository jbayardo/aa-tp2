from abc import abstractmethod

import numpy as np

from rl.agent.baseagent import BaseAgent
from rl.environment.baseenvironment import BaseEnvironment
from rl.agent.evaluationqagent import EvaluationQAgent
from rl.agent.qtable import QTable


class QAgent(BaseAgent):
    def __init__(self, learning_rate, discount_factor, q: QTable = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if q is None:
            q = QTable()
        self._q = q

        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    def policy(self, environment: BaseEnvironment, **kwargs):
        return self._select_learning_action(environment, **kwargs)

    def feedback(self, previous: BaseEnvironment, action, current: BaseEnvironment, **kwargs) -> None:
        episode = kwargs['episode']
        turns = kwargs['turns']

        # Compute the reward, learning rate and discount factor for this update
        reward = self._reward(previous, action, current)
        learning_rate = self._learning_rate(self, episode, turns)
        discount_factor = self._discount_factor(self, episode, turns)

        # Estimate the expectation of the maximum
        actions = current.actions
        expectation_estimator = 0.0
        if len(actions) > 0:
            expectation_estimator = np.amax([self._q(current, action) for action in actions])

        # Update the Q table
        previous_state_current_q = self._q(previous, action)
        self._q.update(previous, action,
                       previous_state_current_q + learning_rate * (
                               reward + discount_factor * expectation_estimator - previous_state_current_q))

    def epoch_statistics(self):
        return None

    def episode_statistics(self):
        return self._q.statistics()

    @abstractmethod
    def _reward(self, previous: BaseEnvironment, action, current: BaseEnvironment) -> np.float64:
        raise NotImplementedError()

    @abstractmethod
    def _select_learning_action(self, environment: BaseEnvironment, **kwargs):
        raise NotImplementedError()

    def deploy(self) -> EvaluationQAgent:
        return EvaluationQAgent(self._q, identifier=self.identifier, name=self.name)


# TODO: mark class as abstract?
class DoubleQAgent(QAgent):
    def __init__(self, q2: QTable = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if q2 is None:
            q2 = QTable()
        self._q2 = q2

    def feedback(self, previous: BaseEnvironment, action, current: BaseEnvironment, **kwargs) -> None:
        episode = kwargs['episode']
        turns = kwargs['turns']

        # Compute the reward, learning rate and discount factor for this update
        reward = self._reward(previous, action, current)
        learning_rate = self._learning_rate(self, episode, turns)
        discount_factor = self._discount_factor(self, episode, turns)

        # Select which Q table to use for this action
        A = self._q
        B = self._q2
        choice = np.random.randint(2)
        if choice == 1:
            A, B = B, A

        # Estimate the expectation of the maximum
        actions = current.actions
        expectation_estimator = 0.0
        if len(actions) > 0:
            best_action_by_A = np.argmax([A(current, action) for action in actions])
            expectation_estimator = B(current, actions[best_action_by_A])

        # Update the chosen Q table
        previous_state_current_q = A(previous, action)
        A.update(previous, action,
                 previous_state_current_q + learning_rate * (
                         reward + discount_factor * expectation_estimator - previous_state_current_q))
