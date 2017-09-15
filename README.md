Reinforcement Learning Pseudo-Library

This is an in-development sort-of-library for Reinforcement Learning. It is similar in style to keras-rl (https://github.com/matthiasplappert/keras-rl). Currently, it works with OpenAI's Gym (for environments) and Keras (for models).

Dependencies:
- Python >= 3.5
- Keras >= 2.07
- OpenAI Gym
- Tensorflow >= 1.21 (GPU access for faster training)

How to use:
- Look at example.py for usage case

Algorithms Currently Implemented:
- DQN (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  - Double functionality (https://arxiv.org/abs/1509.06461)

Algorithms To add:
- Completed dueling functionality for DQN (https://arxiv.org/abs/1511.06581)
- DDPG (https://arxiv.org/pdf/1509.02971)
- A3C (https://arxiv.org/abs/1602.01783)
- TRPO (https://arxiv.org/abs/1502.05477)
