import uuid

import matplotlib.pyplot as plt
import pandas
import itertools
import concurrent.futures

from learning_mechanism import LearningMatch
from abstract.random_agent import *
from impl.four_row_agent import *
from utils import *
import os


def generate_parameter_string(params):
    output = str(params['identifier']) + ': ' + params['__class__'] + '\n    '

    for key in params:
        if key == 'identifier' or key == '__class__':
            continue

        output += key
        output += ': '

        if callable(params[key]):
            output += params[key].__name__
        else:
            output += str(params[key])

        output += '\n    '

    output = output[:-2]
    return output


def dump_statistics(file_identifier: str, file_type: str, statistics, left_params, right_params):
    directory = 'results'
    if not os.path.exists(directory):
        os.mkdir(directory)

    df = pandas.DataFrame.from_records(statistics, index='episode_number')
    df.to_csv('{0}_{1}.csv'.format(directory + '/' + file_identifier, file_type))

    left_player_name = 'Player ' + str(left_params['identifier'])
    right_player_name = 'Player ' + str(right_params['identifier'])

    df = []
    for entry in statistics:
        tied = 0.0
        won = 0.0
        loss = 0.0

        if entry['winner'] == left_params['identifier']:
            won = 1.0
            loss = 0.0
        elif entry['winner'] == right_params['identifier']:
            won = 0.0
            loss = 1.0
        else:
            tied = 1.0

        df.append({
            'episode_number': entry['episode_number'],
            left_player_name: won,
            right_player_name: loss,
            'Ties': tied
        })

    df = pandas.DataFrame.from_records(df, index='episode_number')
    df = df[[left_player_name, 'Ties', right_player_name]].rolling(window=500).mean()
    axes = df.plot(yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('Episode Number')
    plt.ylabel('Avg. Wins Over 500 Matches')
    plt.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.3)
    plt.figtext(0.01, 0.05, generate_parameter_string(left_params), fontsize='large')
    plt.figtext(0.5, 0.05, generate_parameter_string(right_params), fontsize='large')
    axes.get_figure().savefig('{0}_{1}.svg'.format(directory + '/' + file_identifier, file_type))


def emulate_match(params):
    (left_data, right_data) = params
    (left_class, left_parameters) = left_data
    (right_class, right_parameters) = right_data
    del right_data, left_data

    left_parameters = left_parameters.copy()
    left_parameters['identifier'] = 0
    left_agent = left_class(**left_parameters)

    right_parameters = right_parameters.copy()
    right_parameters['identifier'] = 1
    right_agent = right_class(**right_parameters)

    left_parameters['__class__'] = left_class.__name__
    right_parameters['__class__'] = right_class.__name__

    run_id = str(uuid.uuid4())[:8]
    print(run_id, 'Parameters for left agent:', left_parameters)
    print(run_id, 'Parameters for right agent:', right_parameters)

    trainer = LearningMatch(left_agent, right_agent)
    training_statistics, playing_statistics = trainer.train_many_matches(run_id, NUMBER_OF_MATCHES)
    dump_statistics(run_id, 'training', training_statistics, left_parameters, right_parameters)
    dump_statistics(run_id, 'playing', playing_statistics, left_parameters, right_parameters)

if __name__ == '__main__':
    plt.style.use('ggplot')

    agents = []

    agents.append((RandomAgent, {}))

    agents.append((EpsilonGreedyFourRowAgent, {
        'epsilon': np.array([0.1, 0.3, 0.6, 0.9]),
        'learning_rate': [turn_decay_50],
        'discount_factor': [const_08]
    }))

    agents.append((FourRowAgent, {
        'learning_rate': [turn_decay_50],
        'discount_factor': [const_08]
    }))

    agents.append((SoftmaxFourRowAgent, {
        'learning_rate': [turn_decay_50],
        'discount_factor': [const_08],
        'temperature': [temperature]
    }))

    # Generate all possible instances for every agent
    preinstantiated_agents = []
    for (agent, parameters) in agents:
        for combination in [dict(zip(parameters, x)) for x in itertools.product(*parameters.values())]:
            preinstantiated_agents.append((agent, combination))

    # Actually run the matches
    NUMBER_OF_MATCHES = 12000

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
    for data in itertools.combinations(preinstantiated_agents, 2):
        executor.submit(emulate_match, data)
