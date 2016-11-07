from abstract.agent import Agent
from impl.four_row_agent import FourRowAgent
from impl.four_row_state import FourRowState


class LearningMatch(object):
    def __init__(self, player_a: Agent, player_b: FourRowAgent):
        self.teaching_player = player_a
        self.learning_player = player_b
        assert self.teaching_player.identifier != self.learning_player.identifier

    def train_single_match(self, learning_rate: float, discount_factor: float) -> FourRowState:
        turn = 0
        board = FourRowState.generate()

        while not board.is_terminal:
            if turn == 0:
                action = self.teaching_player.policy(board)
                board = board.execute(self.teaching_player, action)
            else:
                action = self.learning_player.learning_step(board, learning_rate, discount_factor)
                board = board.execute(self.learning_player, action)

            turn = (turn + 1) % 2

        return board

    def train_many_matches(self, matches: int, learning_rate: float, discount_factor: float) -> FourRowAgent:
        for match in range(matches):
            self.train_single_match(learning_rate, discount_factor)
            # See who won and plot

        return self.learning_player
