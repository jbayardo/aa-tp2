from abstract.agent import Agent
from abstract.state import State
import collections
import numpy as np


class QAgent(Agent):
    def __init__(self, identifier: int):
        super(QAgent, self).__init__(identifier)
        self._q_definition = None
        self._q_initialize()
        self._learning = True

    def name(self):
        return 'Q Learning Agent'

    @property
    def learning(self):
        return self._learning

    def toggle_learning(self):
        self._learning = not self._learning

    def policy(self, state: State):
        assert not state.is_terminal

        if self._learning:
            return self._select_learning_action(state)
        else:
            available_actions = state.actions
            scores = [self._q(state, action) for action in available_actions]

            sorted_actions = sorted(zip(available_actions, scores), key=lambda x: x[1])
            reference_score = sorted_actions[len(sorted_actions) - 1][1]
            selected_actions = []

            while len(sorted_actions) > 0:
                (action, score) = sorted_actions.pop()

                if score < reference_score:
                    break

                selected_actions.append(action)

            return self._select_action_from_best(state, selected_actions)

    def feedback(self, previous_state: State, executed_action, new_state: State, learning_rate: float = 0.9,
                 discount_factor: float = 0.1):
        assert 0.0 <= learning_rate
        assert learning_rate <= 1.0
        assert 0.0 <= discount_factor
        assert discount_factor <= 1.0
        assert self._learning

        reward = self._reward(previous_state, executed_action, new_state)
        scores = np.array([self._q(new_state, action) for action in new_state.actions])

        maximum_factor = 0.0
        if len(scores) > 0:
            maximum_factor = np.amax(scores)

        current_q = self._q(previous_state, executed_action)
        self._q_update(previous_state, executed_action,
                       current_q + learning_rate * (reward + discount_factor * maximum_factor - current_q))

    def _q_initialize(self):
        self._q_definition = collections.defaultdict(lambda: 1.0)

    def _q_update(self, state: State, action, value):
        self._q_definition[(state, action)] = value

    def _q(self, state: State, action) -> np.float64:
        return self._q_definition[(state, action)]

    def _reward(self, state, action, new_state):
        raise NotImplementedError()

    def _select_action_from_best(self, state, actions):
        raise NotImplementedError()

    def _select_learning_action(self, state):
        raise NotImplementedError()
