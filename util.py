# Random utility classes and methods
from enum import Enum


class RunType(Enum):
    RAND_FILL = 1
    TRAIN = 2
    TEST = 3


class AgentType(Enum):
    DQN = 'Deep Q-Network (DQN)'
