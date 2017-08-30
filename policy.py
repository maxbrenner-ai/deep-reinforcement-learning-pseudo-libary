'''
Outline:
- Abstract base class at top
- new class for every policy

Current Work:
- Epsilongreedy one is not cont. friendly cuz of action selection
- find a way, if there is one, to make the eps variables constants
'''

import numpy as np
from enum import Enum
from math import exp


class ReturnActionType(Enum):
    Q_VALS = 1
    PROBS = 2


# Abstract base class
class Policy:
    # You use NotImptmentedError() for abstract methods
    def return_action(self, vals, mode):
        raise NotImplementedError()

    def update(self, step):
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):
    def __init__(self, init_eps=1.0, min_epsilon=0.01, decay=0.001):
        if init_eps < min_epsilon or init_eps > 1.0: # Possblie error here, so might want to make it <= instead of just <
            raise ValueError("`init_eps` should not be less than the min_epsilon or it is greater than 1.0")
        if min_epsilon < 0 or min_epsilon > 1.0:
            raise ValueError("`min_epsilon` should not be less than 0 or greater than 1")
        if decay < 0 or decay > 1.0:
            raise ValueError("`decay` should not be less than 0 or greater than 1")
        self.eps = init_eps
        self.MAX_EPS = init_eps
        self.MIN_EPS = min_epsilon
        self.DECAY = decay

    # This works for both q_vals and probs cuz either way a vector of vals that matches the length of the num of
    # actions should be sent in
    def return_action(self, vals, mode):
        assert vals.ndim == 1  # Make sure its a vector
        nb_actions = vals.shape[0]

        if np.random.uniform() < self.eps:
            return np.random.random_integers(0, nb_actions - 1)

        if mode == ReturnActionType.Q_VALS:
            return np.argmax(vals)
        elif mode == ReturnActionType.PROBS:
            return np.random.choice(nb_actions, p=vals)

    def update(self, step):
        self.eps = self.MIN_EPS + (self.MAX_EPS - self.MIN_EPS) * exp(-self.DECAY * step)
        # print("Eps:", self.eps)


class RandomPolicy(Policy):
    def return_action(self, vals, mode):
        return np.random.random_integers(0, vals - 1)  # doesnt matter if q or prob vals, its random

    def update(self, step):
        pass


class GreedyPolicy(Policy):
    def return_action(self, vals, mode):
        assert vals.ndim == 1
        return np.argmax(vals)  # doesnt matter if q or prob vals, biggest is chosen

    def update(self, step):
        pass
