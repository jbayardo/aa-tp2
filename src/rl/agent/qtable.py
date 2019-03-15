import collections

import numpy as np

from rl.environment.baseenvironment import BaseEnvironment


class QTable(object):
    def __init__(self):
        self._q_definition = {}
        self._q_sum = 0

    def __call__(self, state: BaseEnvironment, action) -> np.float64:
        return self.get(state, action)

    def get(self, state: BaseEnvironment, action) -> np.float64:
        return self._q_definition.get((state, action), 1.0)

    def update(self, state: BaseEnvironment, action, value: np.float64) -> None:
        key = (state, action)
        self._q_sum -= self._q_definition.get(key, 0.0)
        self._q_sum += value
        self._q_definition[key] = value

    def statistics(self):
        if len(self._q_definition) == 0:
            return 0.0

        return self._q_sum / len(self._q_definition)


class TrackingQTable(QTable):
    def __init__(self):
        super().__init__()
        self._update_count = collections.defaultdict(int)
        self._update_diffs = collections.defaultdict(list)

    def update(self, state: BaseEnvironment, action, value: np.float64):
        previous_q = self.get(state, action)

        super().update(state, action, value)

        key = (state, action)
        self._update_count[key] += 1
        self._update_diffs[key].append(previous_q - value)
