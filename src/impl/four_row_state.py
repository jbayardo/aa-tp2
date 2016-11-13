import copy
import collections
from typing import List
from abstract.state import State
from impl.four_row_action import FourRowAction


class FourRowState(State):
    # TODO: optimize.
    def __init__(self, rows=8, columns=8):
        self._rows = rows
        self._columns = columns

        self._actions = [FourRowAction(column) for column in range(self._columns)]
        self._filled_up_to = [0] * columns
        self._state = collections.defaultdict(lambda: -1)

        self._last_row_modified = 0
        self._last_column_modified = 0
        self._discs_filled = 0

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def actions(self) -> List[FourRowAction]:
        return self._actions

    @staticmethod
    def generate():
        return FourRowState()

    def print_board(self):
        for row in range(self._rows):
            for column in range(self._columns):
                print(" ", self._state[(row, column)], end=" ")

            print("")

    def execute(self, agent, action):
        if action.column >= self._columns:
            raise ArithmeticError('Attempted insert into unknown column')

        if self._filled_up_to[action.column] >= self._rows:
            raise ArithmeticError('Attempted to execute unavailable action')

        new_state = copy.deepcopy(self)

        new_state._filled_up_to[action.column] += 1
        new_state._actions = [a for a in self._actions if new_state._filled_up_to[a.column] < new_state.rows]
        new_state._state[(self._filled_up_to[action.column], action.column)] = agent.identifier

        new_state._last_row_modified = self._filled_up_to[action.column]
        new_state._last_column_modified = action.column
        new_state._discs_filled += 1

        return new_state

    @property
    def is_terminal(self):
        # I only have to check the row, column and diagonals of the last position modified

        agent_identity = self._state[(self._last_row_modified, self._last_column_modified)]

        # Check if the board is empty
        if agent_identity == -1:
            return False

        finished = False

        # Check if board is filled up
        if self._discs_filled == self._rows * self._columns:
            finished = True

        # Check vertical
        number_of_contiguous_discs = 0
        for row in range(self._rows):

            if self._state[(row, self._last_column_modified)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                # print("Win vertical")
                break

        # Check horizontal
        number_of_contiguous_discs = 0
        for column in range(self._columns):

            if self._state[(self._last_row_modified, column)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                # print("Win horizontal")
                break

        # Checking diagonals from left to right
        diagonal_row_position = self._last_row_modified
        diagonal_column_position = self._last_column_modified

        # Going to the first position of the diagonal
        while diagonal_row_position > 0 and diagonal_column_position > 0:
            diagonal_row_position -= 1
            diagonal_column_position -= 1

        number_of_contiguous_discs = 0
        while diagonal_row_position < self._rows and diagonal_column_position < self._columns:

            if self._state[(diagonal_row_position, diagonal_column_position)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                # print("Win diagonal left to right")
                break

            diagonal_row_position += 1
            diagonal_column_position += 1

        # Checking diagonals from right to left
        diagonal_row_position = self._last_row_modified
        diagonal_column_position = self._last_column_modified

        # Going to the first position of the diagonal
        while diagonal_row_position > 0 and diagonal_column_position < self._columns - 1:
            diagonal_row_position -= 1
            diagonal_column_position += 1

        number_of_contiguous_discs = 0
        while diagonal_row_position < self._rows and diagonal_column_position > 0:

            if self._state[(diagonal_row_position, diagonal_column_position)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                # print("Win diagonal right to left")
                break

            diagonal_row_position += 1
            diagonal_column_position -= 1

        return finished
