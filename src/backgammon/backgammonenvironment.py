import copy

import numpy as np

from rl.environment import Environment

COLORS = {0: 'black', 1: 'white'}


class BackgammonAction(object):
    pass


class EnterAction(BackgammonAction):
    def __init__(self, position: int):
        self.position = position


class MoveAction(BackgammonAction):
    def __init__(self, position: int, distance: int):
        self.position = position
        self.distance = distance


class BackgammonEnvironment(Environment):
    @staticmethod
    def generate() -> 'Environment':
        return BackgammonEnvironment()

    def __init__(self):
        self._board = np.zeros(shape=(24, ), dtype=np.uint8)

        self._removed = {'black': 0, 'white': 0}

        self._eaten = {'black': 0, 'white': 0}

        self._initialize()

        self._throw_dice()

    def _initialize(self):
        self._set('black', 23, 2)
        self._set('black', 12, 5)
        self._set('black', 7, 3)
        self._set('black', 5, 5)

        self._set('white', 23, 2)
        self._set('white', 12, 5)
        self._set('white', 7, 3)
        self._set('white', 5, 5)

    def _throw_dice(self):
        self._dice = np.random.randint(1, 7, 2, dtype=np.uint8)

    def actions(self, agent):
        color = COLORS[agent.identifier]
        return self._actions_for_color(color)

    def _actions_for_color(self, color):
        if self._dice[0] != self._dice[1]:
            alternatives = [
                [self._dice[0], self._dice[1]],
                [self._dice[1], self._dice[0]],
            ]
        else:
            alternatives = [[self._dice[0]] * 4]

        checkers = self._checkers_of_color(color)
        opponent_checkers = self._checkers_of_color(self._opponent(color))

        if self._eaten[color] >= 1:
            entry_available = []
            for position in range(18, 24):
                if self._number_of_checkers(color, position) < -1:
                    continue
                entry_available.append(position)

            pass

        for alternative in alternatives:
            pass

    @property
    def is_terminal(self) -> bool:
        return self._removed['black'] == 15 or self._removed['white'] == 15

    def execute(self, agent: 'Agent', actions) -> 'BackgammonEnvironment':
        color = COLORS[agent.identifier]

        env = copy.deepcopy(self)
        for action in actions:
            if isinstance(action, EnterAction):
                env._enter(color, action.position)
            elif isinstance(action, MoveAction):
                env._move(color, action.position, action.distance)
            else:
                raise ValueError('Invalid action')
        env._throw_dice()
        return env

    # The following are required for dictionary usage

    def __hash__(self):
        return hash((self._board, self._removed, self._eaten))

    def __eq__(self, other: 'BackgammonEnvironment') -> bool:
        return self._board == other._board and self._removed == other._removed and self._eaten == other._eaten

    def __ne__(self, other: 'BackgammonEnvironment') -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def _enter(self, color, position: int):
        assert 18 <= position < 24
        assert self._eaten[color] >= 1

        if self._number_of_checkers(color, position) < -1:
            raise ValueError(
                'Attempt to enter into position %d when there are 2 or more opponent checkers'
                % position)

        self._eaten[color] -= 1
        if self._number_of_checkers(color, position) == -1:
            self._eaten[self._opponent(color)] += 1
            self._set(color, position, 1)
        else:
            self._set(color, position,
                      self._number_of_checkers(color, position) + 1)

    def _move(self, color, position: int, distance: int):
        assert 0 <= position < 24
        assert distance > 0

        if self._eaten[color] > 0:
            raise ValueError(
                'Attempt to move a %s from position %d when there are %d eaten'
                % (color, position, self._eaten[color]))

        if self._number_of_checkers(color, position) <= 0:
            raise ValueError(
                'Attempt to move a %s from position %d when there are none' %
                (color, position))

        new_position = position - distance
        if new_position < 0:
            if not self._remove_allowed(color):
                raise ValueError(
                    'Removal is not allowed and you are going off the board')

            self._removed[color] += 1
            self._set(color, position,
                      self._number_of_checkers(color, position) - 1)
        else:
            if self._number_of_checkers(color, new_position) < -1:
                raise ValueError(
                    'There is more than one opponent at position %d' %
                    new_position)

            if self._number_of_checkers(color, new_position) == -1:
                self._eaten[self._opponent(color)] += 1
                self._set(color, new_position, 1)
            else:
                assert self._number_of_checkers(color, new_position) >= 0
                self._set(color, new_position,
                          self._number_of_checkers(color, new_position) + 1)

            self._set(color, position,
                      self._number_of_checkers(color, position) - 1)

    def _set(self, color, position: int, amount: int):
        assert 0 <= position < 24
        assert amount >= 0
        index = self._index_color(color, position)
        if color == 'white':
            amount = -amount
        self._board[index] = amount

    def _index_color(self, color, position: int) -> int:
        if color == 'white':
            return 23 - position
        else:
            return position

    def _remove_allowed(self, color) -> bool:
        assert color in ['black', 'white']
        if color == 'black':
            range_start = 0
            range_end = 6

            total = 0
            for i in range(range_start, range_end):
                if self._board[i] > 0:
                    total += self._board[i]

            if total + self._removed['black'] == 15:
                return True
        else:
            range_start = 18
            range_end = 24

            total = 0
            for i in range(range_start, range_end):
                if self._board[i] < 0:
                    total -= self._board[i]

            if total + self._removed['white'] == 15:
                return True

        return False

    def _number_of_checkers(self, color, position):
        index = self._index_color(color, position)
        if color == 'black':
            return self._board[index]
        elif color == 'white':
            return -self._board[index]
        raise ValueError('Invalid color %s' % color)

    def _opponent(self, color):
        if color == 'white':
            return 'black'
        return 'white'

    def _checkers_of_color(self, color):
        positions = []
        for position in range(24):
            if self._number_of_checkers(color, position) > 0:
                positions.append(position)
        return positions
