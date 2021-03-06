import copy

import numpy as np

from rl.environment import Environment

_rows = 6
_columns = 7

# Only works for 4 contiguous discs
_rows1 = _rows + 1
_rows2 = _rows + 2

_size = _rows * _columns
# Ensure the actual board size is lower than the maximum than can be held by an uint64
assert (_rows1 * _columns) < 64

# Generate the bitmask for checking if the entire board has been filled; idea
# inspired by https://github.com/nwestbury/pyConnect4/blob/master/board.py
_full_board = np.uint64(0)
for i in range(_columns):
    for j in range(_rows):
        _full_board |= np.uint64(1 << (j + (i * _rows1)))


def _is_winning_board(board) -> bool:
    # taken from http://stackoverflow.com/q/7033165/1524592
    board = int(board)
    y = board & (board >> _rows)
    if y & (y >> 2 * _rows):  # check \ diagonal
        return True
    y = board & (board >> _rows1)
    if y & (y >> 2 * _rows1):  # check horizontal
        return True
    y = board & (board >> _rows2)
    if y & (y >> 2 * _rows2):  # check / diagonal
        return True
    y = board & (board >> 1)
    if y & (y >> 2):  # check vertical
        return True
    return False


class FourRowEnvironment(Environment):
    def __init__(self):
        self._heights = np.zeros(_columns, dtype=np.uint8)
        self._boards = np.zeros(2, dtype=np.uint64)
        self._discs_filled = 0

    @staticmethod
    def columns():
        return _columns

    @staticmethod
    def rows():
        return _rows

    def actions(self, agent: 'FourRowAgent'):
        return [i for i in range(_columns) if self._heights[i] < _rows]

    @staticmethod
    def generate() -> 'FourRowEnvironment':
        return FourRowEnvironment()

    def print_board(self):
        print(self._boards)

        for r in reversed(range(_rows)):
            for c in range(_columns):
                p = False

                for player, board in enumerate(self._boards):
                    move = np.uint64(1 << (r + (c * _rows1)))
                    if board & move:
                        p = True
                        print(player + 1, end=" ")
                        break

                if not p:
                    print(0, end=" ")

            print("")

        print("")

    def execute(self, agent: 'FourRowAgent', action) -> 'FourRowEnvironment':
        assert action < _columns
        assert self._heights[action] < _rows
        assert not self.is_terminal
        assert agent.identifier == 0 or agent.identifier == 1

        move = np.uint64(1 << (self._heights[action] + (action * _rows1)))
        new_state = copy.deepcopy(self)
        new_state._boards[agent.identifier] ^= move
        new_state._discs_filled += 1
        new_state._heights[action] += 1

        return new_state

    @property
    def winner(self):
        if self.is_terminal:
            if _is_winning_board(self._boards[0]):
                return 0

            if _is_winning_board(self._boards[1]):
                return 1

        return -1

    @property
    def board(self):
        return self._boards[0] ^ self._boards[1]

    @property
    def is_draw(self):
        return self.board == _full_board

    def get(self, row: int, column: int) -> int:
        position = np.uint64(1 << (row + (column * _rows1)))

        if self._boards[0] & position > 0:
            if self._boards[1] & position > 0:
                return -1
            return 0

        if self._boards[1] & position > 0:
            return 1

        return -1

    @property
    def is_terminal(self) -> bool:
        return self.is_draw or _is_winning_board(self._boards[0]) or _is_winning_board(self._boards[1])

    # The following are required for dictionary usage
    def __hash__(self):
        return hash((self._boards[0], self._boards[1]))

    def __eq__(self, other: 'FourRowEnvironment') -> bool:
        return self.board == other.board

    def __ne__(self, other: 'FourRowEnvironment') -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)
