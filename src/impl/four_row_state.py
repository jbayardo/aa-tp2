import copy
from abstract.state import State
from impl.four_row_action import FourRowAction
import numpy as np
import hashlib

class FourRowState(State):
    # TODO: optimize.
    def __init__(self, rows: int = 6, columns: int = 7):
        self._contiguous_discs = 4
        self._rows = rows
        self._columns = columns
        self._size = rows * columns

        self._actions = np.array([FourRowAction(column) for column in range(self._columns)])
        self._filled_up_to = [7 * i for i in range(6)]
        self._bitboards = [int(self._size * '0', 2) for i in range(2)] # Bitboard for each player

        self._discs_filled = 0
        self._winner = None

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def actions(self):
        return self._actions

    @staticmethod
    def generate() -> 'FourRowState':
        return FourRowState()

    def print_board(self) -> None:
        learning_player = bin(self._bitboards[0])[::-1][:-2]
        teaching_player = bin(self._bitboards[1])[::-1][:-2]
        learning_player += '0' * (self._size - len(learning_player))
        teaching_player += '0' * (self._size - len(teaching_player))
        for row in range(self._rows):
            for column in range(self._columns):
                if learning_player[7 * row + column] == '1':
                    print("0", end=" ")
                elif teaching_player[7 * row + column] == '1':
                    print("1", end=" ")
                else:
                    print("2", end=" ")

            print("")

        print(" ")

    def execute(self, agent: 'FourRowAgent', action: FourRowAction) -> 'FourRowState':
        if action.column >= self._columns:
            raise ArithmeticError('Attempted insert into unknown column')

        if self._filled_up_to[action.column] >= self._rows:
            raise ArithmeticError('Attempted to execute unavailable action')

        new_state = copy.deepcopy(self)

        move = 1 << new_state._filled_up_to[action.column]
        new_state._filled_up_to[action.column] += 1

        bitboard[agent.identifier] ^= move

        new_state._winner = None
        new_state._discs_filled += 1

        if new_state.is_terminal:
            new_state._actions = []

        return new_state

    def is_win(self, agent=0):
        board = self._bitboards[agent] # Need to receive player

        directions = [1, 6, 7, 8]

        for direction in directions:
            if (board & (board >> direction) & (board >> (2 * direction)) & (board >> (3 * direction))) != 0:
                self._winner = agent
                return True

        return False

    def is_terminal(self, agent=0) -> bool:
        # Check if board is empty
        if self._discs_filled == 0:
            return False

        # Check if board is completely full
        if self._discs_filled == self._rows * self._columns:
            return True

        # Check if anyone won
        return self.is_win(agent)

    # The following are required for dictionary usage
    def __hash__(self):
        return int(hashlib.sha1(self._state).hexdigest(), 16)

    def __eq__(self, other: 'FourRowState') -> bool:
        return np.array_equal(self._state, other._state)

    def __ne__(self, other: 'FourRowState') -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
