from impl.four_row_agent import SoftmaxFourRowAgent, EpsilonGreedyFourRowAgent
from impl.random_agent import RandomAgent
from learning_mechanism import LearningMatch

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    matplotlib.style.use('ggplot')

    learning_player = EpsilonGreedyFourRowAgent(0)
    teaching_player = RandomAgent(1)
    trainer = LearningMatch(teaching_player, learning_player)
    learned_player, statistics = trainer.train_many_matches(100000, 0.001, 0.5)

    data = pd.DataFrame.from_records(statistics, index='episode_number')
    data.to_csv('output.csv')

    data[['avg_q', 'std_q']].plot().get_figure().savefig('q_values.png')
    plt.clf()
    plt.cla()
    plt.close()

    data['turns'].plot().get_figure().savefig('turns_taken.png')

    data = []
    for entry in statistics:
        if entry['winner'] == learning_player.identifier:
            won = 1.0
        else:
            won = 0.0

        if entry['winner'] == -1:
            tied = 1.0
        else:
            tied = 0.0

        data.append({
            'episode_number': entry['episode_number'],
            'won': won,
            'tied': tied
        })

    data = pd.DataFrame.from_records(data, index='episode_number')
    data[['won', 'tied']].rolling(window=500).mean().plot().get_figure().savefig('avgs.png')
