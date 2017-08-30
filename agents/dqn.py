'''
includes double and dueling functionalities

Work on:
- add dueling option
'''
from agents.base import Agent
from policy import ReturnActionType
import random
import numpy as np
from memory import Memory


class DQN(Agent):
    def __init__(self, double_dqn=True, batch_size=32, max_memory_length=1000, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)
        if max_memory_length < 1:
            raise ValueError('`max_memory_length` is < 1.')
        if batch_size < 1 or batch_size > max_memory_length:
            raise ValueError('`batch_size` is < 1 or > `max_memory_length`')
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.memory = Memory(max_memory_length)
        # Pretty sure dont need to compile the target model
        self.beh_model.compile(loss='mse', optimizer=self.optimizer)
        self.uses_replay = True

    def act(self, state): #s.reshape(1, self.stateCnt)
        output = self.beh_model.predict(state)[0]  # assumes all models only use state as input
        return self.currently_used_policy.return_action(output, ReturnActionType.Q_VALS)

    def remember(self, state, action, reward, next_state):
        self.memory.add((state, action, reward, next_state))

    # This updates the beh model params
    def update_params(self, step):
        minibatch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                if not self.double_dqn:
                    target = reward + self.gamma * np.amax(self.tar_model.predict(next_state)[0])
                else:
                    target = reward + self.gamma * self.tar_model.predict(next_state)[0][np.argmax(self.beh_model.predict(next_state)[0])]
            target_f = self.beh_model.predict(state)
            target_f[0][action] = target
            self.beh_model.train_on_batch(state, target_f)
            # self.beh_model.fit(state, target_f, batch_size=self.batch_size, epochs=1, verbose=0)

