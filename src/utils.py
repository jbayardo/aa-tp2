import numpy as np


def turn_decay_50(agent, episode, turn):
    return np.float64(turn)/np.float64(50.0)


def const_08(agent, episode, turn):
    return 0.8


def temperature(agent, episode, turn):
    return 10.0/float(turn + 1)
