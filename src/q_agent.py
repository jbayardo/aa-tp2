from abstract.agent import Agent
from abstract.state import State
import numpy as np


class QAgent(Agent):
    def __init__(self, identifier: int, learning_rate, discount_factor):
        super(QAgent, self).__init__(identifier)
        self._q_definition = None
        self._q_initialize()

        self._learning = True

        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    @property
    def learning(self) -> bool:
        return self._learning

    def toggle_learning(self) -> None:
        self._learning = not self._learning

    def policy(self, state: State):
        assert not state.is_terminal

        if self.learning:
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

    def feedback(self, previous_state: State, executed_action, new_state: State, episode: int, turn: int) -> None:
        assert self.learning

        reward = self._reward(previous_state, executed_action, new_state)
        scores = np.array([self._q(new_state, action) for action in new_state.actions])

        maximum_factor = 0.0
        if len(scores) > 0:
            maximum_factor = np.amax(scores)

        current_q = self._q(previous_state, executed_action)

        learning_rate = self._learning_rate(self, episode, turn)
        discount_factor = self._discount_factor(self, episode, turn)

        # noinspection PyTypeChecker
        self._q_update(previous_state, executed_action,
                       current_q + learning_rate * (reward + discount_factor * maximum_factor - current_q))

    def _q_initialize(self) -> None:
        self._q_definition = {}

    def _q_update(self, state: State, action, value: np.float64) -> None:
        self._q_definition[(state, action)] = value

    def _q(self, state: State, action) -> np.float64:
        return self._q_definition.get((state, action), 1.0)

    def _reward(self, state: State, action, new_state: State) -> np.float64:
        raise NotImplementedError()

    def _select_action_from_best(self, state: State, actions):
        raise NotImplementedError()

    def _select_learning_action(self, state: State):
        raise NotImplementedError()

    def perturb(self):
        for (state, action) in self._q_definition:
            self._q_definition[(state, action)] += np.random.uniform()
