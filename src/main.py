import uuid
import matplotlib.pyplot
import pandas
import itertools
import concurrent.futures

from learning_mechanism import LearningMatch
from abstract.random_agent import *
from impl.four_row_agent import *
from utils import *


def dump_statistics(file_identifier: str, file_type: str, statistics, identifier: int, params):
    data = pandas.DataFrame.from_records(statistics, index='episode_number')
    data.to_csv('{0}_{1}.csv'.format(file_identifier, file_type))

    data = []
    for entry in statistics:
        if entry['winner'] == identifier:
            won = 1.0
        else:
            won = 0.0

        if entry['winner'] == -1:
            tied = 1.0
        else:
            tied = 0.0

        data.append({
            'episode_number': entry['episode_number'],
            'left_won': won,
            'tied': tied
        })

    data = pandas.DataFrame.from_records(data, index='episode_number')
    data = data[['left_won', 'tied']].rolling(window=500).mean()
    axes = data.plot(yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes.text(2, 6, 'Parameters A')
    axes.text(4, 6, 'Parameters B')
    axes.get_figure().savefig('{0}_{1}.svg'.format(file_identifier, file_type))


def emulate_match(params):
    (left_data, right_data) = params
    (left_class, left_parameters) = left_data
    (right_class, right_parameters) = right_data

    run_id = str(uuid.uuid4())[:8]
    print('{} Running match between instance of {} and {}'.format(run_id, left_class.__name__, right_class.__name__))
    print('{} Parameters for left agent:'.format(run_id), left_parameters)
    print('{} Parameters for right agent:'.format(run_id), right_parameters)

    left_parameters.update({
        'identifier': 0
    })
    left_agent = left_class(**left_parameters)

    right_parameters.update({
        'identifier': 1
    })
    right_agent = right_class(**right_parameters)

    trainer = LearningMatch(left_agent, right_agent)
    training_statistics, playing_statistics = trainer.train_many_matches(run_id, NUMBER_OF_MATCHES)
    dump_statistics(run_id, 'training', training_statistics, 0, left_parameters)
    dump_statistics(run_id, 'playing', playing_statistics, 0, right_parameters)

if __name__ == '__main__':
    matplotlib.pyplot.style.use('ggplot')

    agents = []

    agents.append((RandomAgent, {}))

    agents.append((EpsilonGreedyFourRowAgent, {
        'epsilon': np.arange(0.1, 1.0, 0.1),
        'learning_rate': [decaying_learning_rate],
        'discount_factor': [decaying_discount_factor]
    }))

    agents.append((FourRowAgent, {
        'learning_rate': [decaying_learning_rate],
        'discount_factor': [decaying_discount_factor]
    }))

    agents.append((SoftmaxFourRowAgent, {
        'learning_rate': [decaying_learning_rate],
        'discount_factor': [decaying_discount_factor],
        'temperature': [temperature]
    }))

    # Generate all possible instances for every agent
    preinstantiated_agents = []
    for (agent, parameters) in agents:
        for combination in [dict(zip(parameters, x)) for x in itertools.product(*parameters.values())]:
            preinstantiated_agents.append((agent, combination))

    # Actually run the matches
    NUMBER_OF_MATCHES = 20000
    print('Emulating {} matches per run'.format(NUMBER_OF_MATCHES))

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
    for data in itertools.combinations(preinstantiated_agents, 2):
        executor.submit(emulate_match, data)
