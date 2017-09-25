# Reinforcement Learning Pseudo-Library

This is an in-development sort-of-library for Reinforcement Learning. It is similar in style to [keras-rl](https://github.com/matthiasplappert/keras-rl). Currently, it works with OpenAI's Gym (for environments) and Keras (for models).

## Dependencies:
- Python >= 3.5
- Keras >= 2.07
- OpenAI Gym
- Tensorflow >= 1.21 (GPU access for faster training)

## How to use:
You can run and look at this example:
```
python examples/dqn_cartpole.py
```

## Algorithms Currently Implemented:
- DQN [[1](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
- Double DQN [[2](https://arxiv.org/abs/1509.06461)]
- Dueling DQN [[3](https://arxiv.org/abs/1511.06581)]

## Algorithms to add next:
- Prioritized Experience Replay (DQN) [[4](https://arxiv.org/pdf/1511.05952.pdf)]
- Deep Recurrent Q-Learning (DRQN) [[5](https://arxiv.org/pdf/1507.06527.pdf)]
- DDPG [[6](https://arxiv.org/pdf/1509.02971)]
- A3C [[7](https://arxiv.org/abs/1602.01783)]
- TRPO [[8](https://arxiv.org/abs/1502.05477)]

## References
### Papers
1. Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
2. Deep Reinforcement Learning with Double Q-Learning, Hasselt et al., 2015
3. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al., 2015
4. Prioritized Experience Replay, Schaul et al., 2016
5. Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone, 2015
6. Continuous Control with Deep Reinforcement Learning, Lillicrap et al., 2016
7. Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016
8. Trust Region Policy Optimization, Schulman et al., 2015

### Other
- keras-rl
