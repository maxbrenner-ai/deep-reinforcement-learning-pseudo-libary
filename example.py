from agents.dqn import DQN
from policy import EpsilonGreedyPolicy as EGP
from callbacks import *
from benchmark import Benchmark
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import gym

'''
This is an example of how to use these classes in conjunction with gym and keras
'''

# Make the env, use gym
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Make the model, use Keras (Sequential or Functional Model)
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))

optimizer = Adam(lr=1e-2)

# Make the policy and agent
policy = EGP(0.95, 0.01, decay=0.003)
agent = DQN(double_dqn=True, dueling_dqn=False, model=model, optimizer=optimizer, policy=policy, 
            gamma=0.95, target_model_update_policy='soft', target_model_hard_policy_wait=200, 
            target_model_soft_policy_constant=0.9, max_memory_length=1000)

# Make the callbacks
rew_cb = PrintReward()
eps_cb = PrintEpsilon(episodic=True, iterations=None)

# Make Benchmark if you want to save agent test data
benchmark = Benchmark('test_benchmark', episode_iteration=1)

# Tran and Test
agent.train(env, 10000, 500, print_rew_cb=rew_cb, print_eps_cb=eps_cb, visualize=False, allow_printing=True)
agent.test(env, 1000, 500, print_rew_cb=rew_cb, benchmark=benchmark, visualize=True, allow_printing=True)
