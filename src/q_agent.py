from abstract.action import Action
from abstract.state import State
from abstract.agent import Agent
from collections import defaultdict


class QAgent(Agent):
    def __init__(self, identifier: int):
        super(QAgent, self).__init__(identifier)
        self._q_definition = None
        self._q_initialize()

    def name(self):
        return 'Q Learning Agent'

    def policy(self, state):
        assert not state.is_terminal

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

    def _q_initialize(self):
        import random
        self._q_definition = defaultdict(random.random)

    def _q_update(self, state, action, value: float):
        self._q_definition[(state, action)] = value

    def _q(self, state, action):
        return self._q_definition[(state, action)]

    def _reward(self, state, action, new_state):
        raise NotImplementedError()

    def _select_action_from_best(self, state, actions):
        raise NotImplementedError()

    def _select_learning_action(self, state):
        raise NotImplementedError()

    def learning_step(self, oponent, state, learning_rate: float = 0.5, discount_factor: float = 0.1):
        action = self._select_learning_action(state)
        new_state = state.execute(self, action)
        reward = self._reward(state, action, new_state)

        try:
            new_state = new_state.execute(oponent, oponent.policy(new_state))

            try:
                maximum_factor = max([self._q(new_state, action) for action in new_state.actions])
            except ValueError:
                maximum_factor = 0.0

            current_q = self._q(state, action)
            self._q_update(state, action,
                           current_q + learning_rate * (reward + discount_factor * maximum_factor - current_q))
        except:
            pass

        return action
