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

        winner = -1
        if turn == 0 and board.is_terminal:
            #print("Gano el Learning player!")
            winner = self.learning_player.identifier
        elif turn == 1 and board.is_terminal:
            #print("Gano el Teaching player!")
            winner = self.teaching_player.identifier
        else:
            print("Empate de perdedores")

        return board, winner

    def train_many_matches(self, matches: int, learning_rate: float, discount_factor: float) -> FourRowAgent:
        learning_player_matches_won = []
        teaching_player_matches_won = []
        ties = []
        for match in range(matches):
            board, winner = self.train_single_match(learning_rate, discount_factor)

            if winner == self.learning_player.identifier:
                learning_player_matches_won.append(1)
                teaching_player_matches_won.append(0)
            elif winner == self.teaching_player.identifier:
                teaching_player_matches_won.append(1)
                learning_player_matches_won.append(0)
            else:
                ties.append(1)

        print("learning matches won:" , sum(learning_player_matches_won))
        print("teaching matches won:" , sum(teaching_player_matches_won))
        print("ties: " , sum(ties))

        return self.learning_player, learning_player_matches_won, teaching_player_matches_won
