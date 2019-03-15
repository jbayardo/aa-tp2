import random

from rl.agent.baseagent import BaseAgent
from rl.environment.baseenvironment import BaseEnvironment
from rl.agent.qtable import QTable


class EvaluationQAgent(BaseAgent):
    def __init__(self, q: QTable, *args, **kwargs):
        assert q is not None

        super(EvaluationQAgent, self).__init__(*args, **kwargs)
        self._q = q

    def policy(self, environment: BaseEnvironment, **kwargs):
        available_actions = environment.actions
        scores = [self._q(environment, action) for action in available_actions]

        sorted_actions = sorted(zip(available_actions, scores), key=lambda x: x[1])
        reference_score = sorted_actions[len(sorted_actions) - 1][1]
        selected_actions = []

        while len(sorted_actions) > 0:
            (action, score) = sorted_actions.pop()

            if score < reference_score:
                break

            selected_actions.append(action)

        return self._select_action_from_best(environment, selected_actions)

    def _select_action_from_best(self, state: BaseEnvironment, actions):
        return random.choice(actions)
