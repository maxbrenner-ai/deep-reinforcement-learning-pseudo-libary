from agents.dqn import DQN
from policy import EpsilonGreedyPolicy as EGP
from keras.models import Sequential
from keras.layers import *
import gym
from callbacks import *
from benchmark import Benchmark
import tensorflow as tf
from agents.dqn import PER
from util import StateProcessor

# First create the environment and get the state shape and action size (number of actions allowed)
env = gym.make('Pong-v0')
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Create a state processor, this will allow you to resize, recolor (in grayscale) and add a screen stack function
state_processor = StateProcessor(state_shape, new_height=84, new_width=84, screen_stack=4, colors='gray')
# Get new state shape for model input
state_shape = state_processor.state_shape

# Then make the model, Sequential and Functional both work
# The data_format for each Conv2D layer needs to be 'channels_last' (this is the default behavior of keras)
model = Sequential()
model.add(Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=state_shape))
model.add(Conv2D(64, 4, 4, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(action_size))

# Learning rate and Optimizer (Must be TF!)
LEARNING_RATE = 1e-3
tf_optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

# Create the policy, this is Epsilon-Greedy (exponential decay)
policy = EGP(init_eps=0.95, min_epsilon=0.01, decay=0.003)

# Can create an object for Prioritized Experience Replay
per = PER(priority_importance=0.6, initial_anneal=0.5, anneal_growth_rate=0.00008)

# Make the agent! In this case a Double DQN with PER
# Can make it dueling and auto add streams, or like the commented model above just dueling and set add streams to false
agent = DQN(double_dqn=True, PER=per, dueling_dqn=False, add_dueling_streams=False, model=model, optimizer=tf_optimizer,
            policy=policy, action_size=action_size, state_processor=state_processor, gamma=0.95,
            target_model_update_policy='soft', target_model_hard_policy_wait=500, target_model_soft_policy_constant=0.9,
            replay_period_wait=4, reward_clipping=True, huber_loss=True, batch_size=64, max_memory_length=10000)

# Make callbacks if you want, reward and epsilon are implemented
rew_cb = PrintReward()
eps_cb = PrintEpsilon(episodic=True, iterations=None)

# Make a benchmark if you want to keep track of info and data on the agents testing performance
benchmark = Benchmark('bench_0', episode_iteration=1)

agent.train(env, 100000, None, print_rew_cb=rew_cb, print_eps_cb=eps_cb, visualize=False, allow_printing=True)

agent.test(env, 50000, None, print_rew_cb=rew_cb, benchmark=benchmark, visualize=False, allow_printing=True)