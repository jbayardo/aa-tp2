import numpy as np

def decaying_learning_rate(agent, episode, turn):
    return np.float64(turn)/np.float64(50.0)


def decaying_discount_factor(agent, episode, turn):
    return 0.8


def temperature(agent, episode, turn):
    return 10.0/float(turn + 1)