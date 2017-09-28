from agents.dqn import DQN
from policy import EpsilonGreedyPolicy as EGP
from callbacks import *
from benchmark import Benchmark
from keras.models import Sequential, Model
from keras.layers import *
import tensorflow as tf
import gym
from agents.dqn import PER

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

# Must use TF optimizers
LEARNING_RATE = 1e-2
tf_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# Make the policy and agent
policy = EGP(init_eps=0.95, min_epsilon=0.01, decay=0.003)

# If you want to use Prioritized Experience Replay can make a PER object or send in None for the PER arg. in the agent
per = PER(priority_importance=0.6, initial_anneal=0.5, anneal_growth_rate=0.00008)

# Make the Agent: Currently support DQN with the optional additions of Double, Dueling and Prioritized ER
# If you make a model with two separate output streams for state value and action values just set 'dueling_dqn' to True
# and set 'add_dueling_streams' to False, of you want dueling and made a normal model (no 2 streams) set them both to T

# This will make a Double Dueling DQN with Prioritized Experience Replay
agent = DQN(double_dqn=True, PER=per, dueling_dqn=True, add_dueling_streams=True, model=model, optimizer=tf_optimizer,
            policy=policy, state_dim=state_dim, action_size=action_size, gamma=0.95, target_model_update_policy='soft',
            target_model_hard_policy_wait=500, target_model_soft_policy_constant=0.9, reward_clipping=True,
            batch_size=32, max_memory_length=1000)

# Make the callbacks
rew_cb = PrintReward()
eps_cb = PrintEpsilon(episodic=True, iterations=None)

# Make Benchmark if you want to save agent test data
benchmark = Benchmark('test_benchmark', episode_iteration=1)

# Tran and Test
agent.train(env, 10000, None, print_rew_cb=rew_cb, print_eps_cb=eps_cb, visualize=False, allow_printing=True)
agent.test(env, 5000, None, print_rew_cb=rew_cb, benchmark=benchmark, visualize=True, allow_printing=True)
