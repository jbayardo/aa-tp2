from abstract.agent import Agent
from impl.four_row_agent import EpsilonGreedyFourRowAgent
from impl.four_row_state import FourRowState


class LearningMatch(object):
    def __init__(self, teaching_agent: Agent, student_player: EpsilonGreedyFourRowAgent):
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

        winner = -1
        if turn == 0 and board.is_terminal:
            winner = self.learning_player.identifier
        elif turn == 1 and board.is_terminal:
            winner = self.teaching_player.identifier

        return winner, turns

    def train_many_matches(self, episodes: int, learning_rate: float, discount_factor: float) -> EpsilonGreedyFourRowAgent:
        from collections import defaultdict
        won_by = defaultdict(int)
        samples = []

        for episode in range(1, episodes + 1):
            winner, turns = self.train_single_match(learning_rate, discount_factor)
            won_by[winner] += 1

            samples.append({
                'episode_number': episode,
                'won_by': winner,
                'avg_by_teacher': float(won_by[2])/float(episode),
                'avg_by_student': float(won_by[1])/float(episode),
                'avg_ties': float(won_by[-1])/float(episode)
            })

        return self.learning_player, samples
