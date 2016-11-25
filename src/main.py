from impl.four_row_agent import SoftmaxFourRowAgent, EpsilonGreedyFourRowAgent
from impl.random_agent import RandomAgent
from learning_mechanism import LearningMatch
import numpy as np
import uuid


def decaying_learning_rate(agent, episode, turn):
    return np.float64(turn)/np.float64(50.0)


def decaying_discount_factor(agent, episode, turn):
    return 0.8


def temperature(agent, call):
    return 1.0/float(call)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    matplotlib.style.use('ggplot')

    run_id = uuid.uuid4()

    learning_player = SoftmaxFourRowAgent(
        identifier=0,
        learning_rate=decaying_learning_rate,
        discount_factor=decaying_discount_factor,
        temperature=temperature)
    teaching_player = RandomAgent(1)
    trainer = LearningMatch(teaching_player, learning_player)
    learned_player, statistics = trainer.train_many_matches(30000)

    data = pd.DataFrame.from_records(statistics, index='episode_number')
    data.to_csv('{0}.csv'.format(run_id))

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
    data[['won', 'tied']].rolling(window=250).mean().plot(yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).get_figure().savefig('{0}.svg'.format(run_id))