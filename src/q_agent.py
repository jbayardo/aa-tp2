from abstract.action import Action
from abstract.state import State
from abstract.agent import Agent
from collections import defaultdict


class QAgent(Agent):
    def __init__(self, identifier: int):
        super(QAgent, self).__init__(identifier)
        self._q_definition = None
        self._q_initialize()

    def name(self) -> str:
        return 'Q Learning Agent'

    def policy(self, state: State) -> Action:
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

    def _q_initialize(self) -> None:
        import random
        self._q_definition = defaultdict(random.random)

    def _q_update(self, state: State, action: Action, value: float) -> None:
        self._q_definition[(state, action)] = value

    def _q(self, state: State, action: Action) -> float:
        return self._q_definition[(state, action)]

    def _reward(self, state: State, action: Action, new_state: State) -> float:
        raise NotImplementedError()

    def _select_action_from_best(self, state: State, actions: Action) -> Action:
        raise NotImplementedError()

    def _select_learning_action(self, state: State) -> Action:
        raise NotImplementedError()

    def learning_step(self, state: State, learning_rate: float = 0.5, discount_factor: float = 0.1) -> Action:
        action = self._select_learning_action(state)
        new_state = state.execute(self, action)
        reward = self._reward(state, action, new_state)

        try:
            maximum_factor = max([self._q(new_state, action) for action in new_state.actions])
        except ValueError:
            maximum_factor = 0.0

        current_q = self._q(state, action)
        self._q_update(state, action,
                       current_q + learning_rate * (reward + discount_factor * maximum_factor - current_q))

        return action
