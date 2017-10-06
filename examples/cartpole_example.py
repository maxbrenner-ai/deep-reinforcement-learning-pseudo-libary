from agents.dqn import DQN
from policy import EpsilonGreedyPolicy as EGP
from keras.models import Model
from keras.layers import *
import gym
from callbacks import *
from benchmark import Benchmark
import tensorflow as tf
from util import StateProcessor

# First create the environment and get the state shape and action size (number of actions allowed)
env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Create a state processor, however its not really used for cartpole, its used for envs with image states
# So in this case, it doesnt matter what you send in other than the state shape, but it is still necessary
state_processor = StateProcessor(state_shape)
# Get the state shape to use for the model input (even though with this env, once again, it doesn't change the state)
state_shape = state_processor.state_shape

# Then make the model, Sequential and Functional both work
input_layer = Input(shape=state_shape)
first_layer = Dense(24, activation='relu')(input_layer)
second_layer = Dense(24, activation='relu')(first_layer)
third_layer = Dense(action_size, activation='linear')(second_layer)
model = Model(input=input_layer, output=third_layer)

# You could also use this if you wanted dueling to be on, but not auto add the streams behind the scenes
# input_layer = Input(shape=(state_dim,))
# first_layer = Dense(128, activation='relu')(input_layer)
# second_layer = Dense(128, activation='relu')(first_layer)
# adv_stream = Dense(action_size, activation='linear')(second_layer)
# state_stream = Dense(1, activation='linear')(second_layer)
# model = Model(input=input_layer, outputs=[adv_stream, state_stream])

# Learning rate and Optimizer (Must be TF!)
LEARNING_RATE = 1e-3
tf_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# Create the policy, this is Epsilon-Greedy (exponential decay)
policy = EGP(init_eps=0.95, min_epsilon=0.01, decay=0.003)

# Make the agent! In this case a Dueling Double DQN (Dueling just for example, for cartpole it doesn't work too well)
# Can use the commented model above that already has streams and set `add_dueling_streams` to False
agent = DQN(double_dqn=True, PER=None, dueling_dqn=True, add_dueling_streams=True, model=model, optimizer=tf_optimizer,
            policy=policy, action_size=action_size, state_processor=state_processor, gamma=0.95,
            target_model_update_policy='soft', target_model_hard_policy_wait=500, target_model_soft_policy_constant=0.9,
            replay_period_wait=1, reward_clipping=True, huber_loss=False, batch_size=32, max_memory_length=1000)

# Make callbacks if you want, reward and epsilon are implemented
rew_cb = PrintReward()
eps_cb = PrintEpsilon(episodic=True, iterations=None)

# Make a benchmark if you want to keep track of info and data on the agents testing performance
benchmark = Benchmark('bench_agent0', episode_iteration=1)

agent.train(env, 10000, None, print_rew_cb=rew_cb, print_eps_cb=eps_cb, visualize=False, allow_printing=True)

agent.test(env, 5000, None, print_rew_cb=rew_cb, benchmark=benchmark, visualize=True, allow_printing=True)