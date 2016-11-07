import copy
from abstract.state import State
from impl.four_row_action import FourRowAction


class FourRowState(State):
    # TODO: optimize.
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.filled_up_to = [0 for x in range(columns)]
        self.state = {}

        for i in range(rows):
            for j in range(columns):
                self.state[(i, j)] = -1

    @property
    def actions(self):
        output = []
        if not self.is_terminal:
            for column in range(self.columns):
                if self.filled_up_to[column] < self.rows:
                    output.append(FourRowAction(column))

        return output

    @staticmethod
    def generate():
        # TODO: This always generates the initially empty state
        output = FourRowState()
        return output

    def execute(self, agent, action):
        if action.column >= self.columns:
            raise ArithmeticError('Attempted insert into unknown column')

        if self.filled_up_to[action.column] >= self.rows:
            raise ArithmeticError('Attempted to execute unavailable action')

        new_state = copy.deepcopy(self)
        new_state.state[(self.filled_up_to[action.column], action.column)] = agent.identifier
        new_state.filled_up_to[action.column] += 1
        return new_state

    @property
    def is_terminal(self) -> bool:
        is_full = True
        for column in range(self.columns):
            if self.filled_up_to[column] < self.rows:
                is_full = False
                break

        if is_full:
            return True

        # TODO: Check if anyone won (i.e. anything repeated in row, column, or diagonal).
        return False
