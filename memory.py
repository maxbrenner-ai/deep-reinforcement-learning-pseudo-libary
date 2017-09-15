from collections import deque
import random


class Memory:
    def __init__(self, max_length):
        self.storage = deque(maxlen=max_length)

    def add(self, element):
        self.storage.append(element)

    # Sample will return a matrix
    def sample(self, amount):
        return random.sample(self.storage, amount)

    def is_full(self):
        if len(self.storage) == self.storage.maxlen:
            return True
        return False
