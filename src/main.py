import concurrent.futures
import itertools
import logging
import os
import pickle
import time

import settings
from four_row.fourrowenvironment import FourRowEnvironment
from gym import Gym
from turnbasedmatch import TurnBasedMatch


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


class ExperimentFilter(logging.Filter):
    def __init__(self, run_version, experiment_id):
        super().__init__()

        self._run_version = run_version
        self._experiment_id = experiment_id

    def filter(self, record):
        record.run_version = self._run_version
        record.experiment_id = self._experiment_id
        return True


def emulate(parameters):
    (left_data, right_data) = parameters
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

    # The experiment_id is supposed to be unique for every parameter set possible. The idea is that this identifies
    # uniquely what the experiment is.
    experiment_id = str(hash((hash(frozenset(left_parameters.items())), hash(frozenset(right_parameters.items())))))

    results_path = os.path.join(os.getcwd(), experiment_id)
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, 'parameters.pickle'), 'wb') as parameters_file_handle:
        pickle.dump(parameters, parameters_file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_log_handler = logging.FileHandler(os.path.join(results_path, 'events.log'))
    file_log_handler.setLevel(logging.DEBUG)

    stream_log_handler = logging.StreamHandler()
    stream_log_handler.setLevel(logging.INFO)
    stream_log_handler.setFormatter(logging.Formatter("%(asctime)-15s %(message)s"))

    log = logging.getLogger(experiment_id)
    log.setLevel(logging.DEBUG)

    log.addHandler(file_log_handler)
    log.addHandler(stream_log_handler)
    log.addFilter(ExperimentFilter(run_version, experiment_id))

    log.info('Experiment %s start. Results at %s', experiment_id, results_path)
    gym = Gym(FourRowEnvironment, TurnBasedMatch, left_agent, right_agent)
    samples = gym.train(experiment_id, results_path,
                        settings.EPOCHS, settings.TRAINING_EPISODES_PER_EPOCH, settings.VALIDATION_EPISODES_PER_EPOCH)
    log.info('Experiment %s end. Results at %s', experiment_id, results_path)

    samples_file_path = os.path.join(results_path, 'samples.pickle')
    with open(samples_file_path, 'wb') as samples_file_handle:
        pickle.dump(samples, samples_file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    log.debug('Samples file has been stored at %s', samples_file_path)


if __name__ == '__main__':
    # The run_version is a unique identifier given to each experiment. The important property about it is that it
    # monotonically increases with time, so that results from two consecutive runs can be evaluated.
    run_version = str(time.monotonic())

    # Set the path into which we will dump our all files
    experiment_path = os.path.join(settings.EXPERIMENT_PATH, run_version)
    os.makedirs(experiment_path, exist_ok=True)
    os.chdir(experiment_path)

    # Actually run the matches
    flights = itertools.combinations(settings.FLIGHTS, 2)

    if settings.PARALLEL_RUN:
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.PARALLEL_WORKERS) as executor:
            for flight in flights:
                executor.submit(emulate, flight)
    else:
        for flight in flights:
            emulate(flight)
