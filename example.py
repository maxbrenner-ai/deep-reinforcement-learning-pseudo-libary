from agents.dqn import DQN
from policy import EpsilonGreedyPolicy as EGP
from callbacks import *
from benchmark import Benchmark
from keras.models import Sequential, Model
from keras.layers import *
import tensorflow as tf
import gym

'''
This is an example of how to use these classes in conjunction with gym and keras
'''

# Make the env, use gym
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_size = env.action_space.n

# Make the model, use Keras (Sequential or Functional Model)
input_layer = Input(shape=(state_dim,))
first_layer = Dense(24, activation='relu')(input_layer)
second_layer = Dense(24, activation='relu')(first_layer)
third_layer = Dense(action_size, activation='linear')(second_layer)
model = Model(input=input_layer, output=third_layer)

# IMPORTANT: For now must use TF optimizers
LEARNING_RATE = 1e-2
tf_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# Make the policy and agent
policy = EGP(0.95, 0.01, decay=0.003)
# This will make a dueling dqn AND auto add dueling streams to the arch. you send in
agent = DQN(double_dqn=True, dueling_dqn=True, add_dueling_streams=True, model=model, optimizer=tf_optimizer,
            policy=policy, state_dim=state_dim, action_size=action_size, gamma=0.95, target_model_update_policy='soft',
            target_model_hard_policy_wait=200, target_model_soft_policy_constant=0.9, reward_clipping=True,
            max_memory_length=1000)

# Make the callbacks
rew_cb = PrintReward()
eps_cb = PrintEpsilon(episodic=True, iterations=None)

# Make Benchmark if you want to save agent test data
benchmark = Benchmark('test_benchmark', episode_iteration=1)

# Tran and Test
agent.train(env, 10000, None, print_rew_cb=rew_cb, print_eps_cb=eps_cb, visualize=False, allow_printing=True)
agent.test(env, 5000, None, print_rew_cb=rew_cb, benchmark=benchmark, visualize=True, allow_printing=True)
