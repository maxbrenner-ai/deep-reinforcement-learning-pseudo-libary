'''
Outline:
- Abstract base class at top
- new class for every policy

Current Work:
- update for epsilongreedy
- Epsilongreedy one is not cont. friendly cuz of action selection
'''

import numpy as np


# Abstract base class
class Policy:
    # You use NotImptmentedError() for abstract methods
    def return_action(self, **args):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):
    def __init__(self, init_eps=1.0):
        self.eps = init_eps

    # This works for both q_vals and probs cuz either way a vector of vals that matches the length of the num of
    # actions should be sent in
    def return_action(self, vals, mode):
        assert mode == 'q_values' or mode == 'probabilities'
        assert vals.ndim == 1  # Make sure its a vector
        nb_actions = vals.shape[0]

        if np.random.uniform() < self.eps:
            return np.random.random_integers(0, nb_actions - 1)

        if mode == 'q_values':
            return np.argmax(vals)

        return np.random.choice(nb_actions, p=vals)

    # Implement this
    def update(self):
        i = 4


class GreedyPolicy(Policy):
    def return_action(self, q_vals):
        assert q_vals.ndim == 1
        return np.argmax(q_vals)

    def update(self):
        pass
