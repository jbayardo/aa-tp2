from impl.four_row_agent import FourRowAgent
from impl.four_row_random_agent import FourRowRandomAgent
from learning_mechanism import LearningMatch

if __name__ == '__main__':
    learning_player = FourRowAgent(1)
    teaching_player = FourRowRandomAgent(2)
    trainer = LearningMatch(teaching_player, learning_player)
    trainer.train_many_matches(10, 0.9, 0.1)
