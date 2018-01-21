import itertools

from rl.randomagent import *
from four_row.fourrowagent import *
from utils import *

EPOCHS = 1
TRAINING_EPISODES_PER_EPOCH = 1000
VALIDATION_EPISODES_PER_EPOCH = 100

EPISODE_LOG_EVERY_N = 100

PARALLEL_RUN = False
PARALLEL_WORKERS = 4
EXPERIMENT_PATH = 'results'

agents = [
    (RandomAgent, {}),
    (EpsilonGreedyFourRowAgent, {
        'epsilon': np.array([0.1, 0.3, 0.6, 0.9]),
        'learning_rate': [turn_decay_50],
        'discount_factor': [const_08]
    }),
    (FourRowAgent, {
        'learning_rate': [turn_decay_50],
        'discount_factor': [const_08]
    }),
    (SoftmaxFourRowAgent, {
        'learning_rate': [turn_decay_50],
        'discount_factor': [const_08],
        'temperature': [temperature]
    })]

# Generate all possible instances for every agent
FLIGHTS = []
for (agent, parameters) in agents:
    for combination in [dict(zip(parameters, x)) for x in itertools.product(*parameters.values())]:
        FLIGHTS.append((agent, combination))

del agents
