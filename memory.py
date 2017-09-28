from collections import deque
import random
from util import SumTree
import numpy.random as rand


# Abstract class
class Memory:
    def add(self, sample, error):
        raise NotImplementedError

    def sample_batch(self, batch_size):
        raise NotImplementedError
        # Make sure returns two lists

    def update(self, *args):
        raise NotImplementedError
        # Make sure returns []

    def is_full(self):
        raise NotImplementedError

# This is just for Experience Replay
class DequeMemory(Memory):
    def __init__(self, max_length):
        self.storage = deque(maxlen=max_length)

    def add(self, sample, error=None):
        assert error is None
        self.storage.append(sample)

    # Sample will return a matrix
    def sample_batch(self, amount):
        return random.sample(self.storage, amount), []

    def update(self, *args):
        return []

    def is_full(self):
        if len(self.storage) == self.storage.maxlen:
            return True
        return False

class SumTreeMemory(Memory):
    def __init__(self, max_length, priority_importance, edge_add):
        self.storage = SumTree(max_length)
        self.a = priority_importance
        self.e = edge_add
        self.current_max_priority = None
        self.current_min_priority = None

    def get_priority(self, error):
        return (error + self.e) ** self.a

    def update_max_prior(self, num):
        self.current_max_priority = max(self.current_max_priority, num) if self.current_max_priority is not None \
            else num

    def update_min_prior(self, num):
        self.current_min_priority = min(self.current_min_priority, num) if self.current_min_priority is not None \
            else num

    # Send in None for error to use the current max prior.
    def add(self, sample, error):
        if error is None:
            assert self.current_max_priority is not None, "Max prior. needs to be set before using it"
            p = self.current_max_priority
        else:
            p = self.get_priority(error)
            self.update_max_prior(p)
            self.update_min_prior(p)
        self.storage.add(p, sample)

    def sample_batch(self, batch_size):
        data_batch = []
        index_batch = []
        segment = self.storage.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (index, p, data) = self.storage.get(s)
            data_batch.append(data)
            index_batch.append(index)

        return data_batch, index_batch

    def update(self, index_batch, error_batch):
        new_prior = []
        size = len(index_batch)
        assert size is len(error_batch)
        for i in range(size):
            p = self.get_priority(error_batch[i])
            new_prior.append(p)
            self.update_max_prior(p)
            self.update_min_prior(p)
            self.storage.update(index_batch[i], p)
        return new_prior

    def is_full(self):
        return self.storage.is_full