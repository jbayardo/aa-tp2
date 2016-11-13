from impl.four_row_agent import EpsilonGreedyFourRowAgent
from impl.four_row_random_agent import FourRowRandomAgent
from learning_mechanism import LearningMatch

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    matplotlib.style.use('ggplot')

    learning_player = EpsilonGreedyFourRowAgent(1)
    teaching_player = FourRowRandomAgent(2)

    trainer = LearningMatch(teaching_player, learning_player)
    learned_player, statistics = trainer.train_many_matches(10000, 0.3, 0.7)
    data = pd.DataFrame.from_records(statistics, index='episode_number')

    plot = data[['avg_by_student', 'avg_by_teacher', 'avg_ties']].plot()
    plot.get_figure().savefig('output.png')
