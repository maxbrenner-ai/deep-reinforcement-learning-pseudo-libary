# Random utility classes and methods
from enum import Enum

# def save_trained
#
# def load_trained_model(file_path):


class RunType(Enum):
    RAND_FILL = 1
    TRAIN = 2
    TEST = 3


class AgentType(Enum):
    DQN = 'Deep Q-Network (DQN)'
