from abstract.agent import Agent
from impl.four_row_agent import EpsilonGreedyFourRowAgent, FourRowAgent
from impl.four_row_state import FourRowState


class LearningMatch(object):
    def __init__(self, teaching_agent: Agent, student_player: FourRowAgent):
        self.teaching_player = teaching_agent
        self.learning_player = student_player
        assert self.teaching_player.identifier != self.learning_player.identifier

    def train_single_match(self, learning_rate: float, discount_factor: float) -> FourRowState:
        board = FourRowState.generate()
        turn = 0
        turns = 0

        while not board.is_terminal:
            if turn == 0:
                action = self.teaching_player.policy(board)
                new_board = board.execute(self.teaching_player, action)
            else:
                action = self.learning_player.learning_step(board, learning_rate, discount_factor)
                new_board = board.execute(self.learning_player, action)

            del board
            board = new_board

            turn = (turn + 1) % 2
            turns += 1

        return board.winner, turns

    def train_many_matches(self, episodes: int, learning_rate: float, discount_factor: float) -> FourRowAgent:
        samples = []

        for episode in range(1, episodes + 1):
            winner, turns = self.train_single_match(learning_rate, discount_factor)

            # Compute average value for the Q
            avgq = 0.0
            totq = 0.0
            for key in self.learning_player._q_definition:
                avgq += self.learning_player._q_definition[key]
                totq += 1.0
            avgq /= totq

            samples.append({
                'episode_number': episode,
                'winner': winner,
                'avg_q': avgq
            })

        return self.learning_player, samples
