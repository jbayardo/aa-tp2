import copy
from abstract.state import State
from impl.four_row_action import FourRowAction


class FourRowState(State):
    # TODO: optimize.
    def __init__(self, rows=8, columns=8):
        self.rows = rows
        self.columns = columns
        self.filled_up_to = [0 for x in range(columns)]
        self.state = {}
        self.last_row_modified = 0
        self.last_column_modified = 0
        self.discs_filled = 0

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


    def print_board(self):
        for row in range(self.rows):
            for column in range(self.columns):
                
                print(" ", self.state[(row, column)], end = " ")

            print("")

    def execute(self, agent, action):
        if action.column >= self.columns:
            raise ArithmeticError('Attempted insert into unknown column')

        if self.filled_up_to[action.column] >= self.rows:
            raise ArithmeticError('Attempted to execute unavailable action')

        new_state = copy.deepcopy(self)
        new_state.state[(self.filled_up_to[action.column], action.column)] = agent.identifier
        new_state.filled_up_to[action.column] += 1
        new_state.last_row_modified = self.filled_up_to[action.column]
        new_state.last_column_modified = action.column
        new_state.discs_filled += 1

        return new_state

    @property
    def is_terminal(self):
        # I only have to check the row, column and diagonals of the last position modified

        agent_identity = self.state[(self.last_row_modified, self.last_column_modified)]

        # Check if the board is empty
        if agent_identity == -1:
            return False

        finished = False

        # Check if board is filled up
        if self.discs_filled == self.rows * self.columns:
          finished = True

        # Check vertical
        number_of_contiguous_discs = 0
        for row in range(self.rows):

            if self.state[(row, self.last_column_modified)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                #print("Win vertical")
                break

        # Check horizontal
        number_of_contiguous_discs = 0
        for column in range(self.columns):

            if self.state[(self.last_row_modified, column)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                #print("Win horizontal")
                break


        # Checking diagonals from left to right
        diagonal_row_position = self.last_row_modified
        diagonal_column_position = self.last_column_modified

        # Going to the first position of the diagonal
        while diagonal_row_position > 0 and diagonal_column_position > 0:
            diagonal_row_position -= 1
            diagonal_column_position -= 1
        
        number_of_contiguous_discs = 0
        while diagonal_row_position < self.rows and diagonal_column_position < self.columns:

            if self.state[(diagonal_row_position, diagonal_column_position)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                #print("Win diagonal left to right")
                break

            diagonal_row_position += 1
            diagonal_column_position += 1


        # Checking diagonals from right to left
        diagonal_row_position = self.last_row_modified
        diagonal_column_position = self.last_column_modified

        # Going to the first position of the diagonal
        while diagonal_row_position > 0 and diagonal_column_position < self.columns - 1:
            diagonal_row_position -= 1
            diagonal_column_position += 1

        number_of_contiguous_discs = 0
        while diagonal_row_position < self.rows and diagonal_column_position > 0:

            if self.state[(diagonal_row_position, diagonal_column_position)] == agent_identity:
                number_of_contiguous_discs += 1
            else:
                number_of_contiguous_discs = 0

            if number_of_contiguous_discs >= 4:
                finished = True
                #print("Win diagonal right to left")
                break

            diagonal_row_position += 1
            diagonal_column_position -= 1

        return finished
