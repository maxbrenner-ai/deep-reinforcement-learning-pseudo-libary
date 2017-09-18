from agents.base import Agent
from policy import ReturnActionType
from memory import Memory
from util import AgentType
from keras import models
from keras.models import Model
from keras.layers import *
import numpy as np


class DQN(Agent):
    def __init__(self, double_dqn=True, dueling_dqn=False, batch_size=32, max_memory_length=1000, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)

        # NOTE: The dueling functionality of the DQN does not work yet ---
        self.dueling_dqn = False
        # if self.dueling_dqn:
        #     self.construct_dueling_streams()
        # ----------------------------------------------------------------

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

        self.agent_type = AgentType.DQN

    # This will take the second to last layer duplicate it, on one stream put the advantage part and the other the state
    def construct_dueling_streams(self):
        # Check that second to last layer is dense if dueling is on
        if type(self.beh_model.layers[-2]) is not Dense:
            raise ValueError('Second to last model layer must be dense for dueling to work')
        # Check to make sure more than two layers
        if len(self.beh_model.layers) < 2:
            raise ValueError('Model needs more than two layers for dueling to work')
        model = models.clone_model(self.beh_model)

        # Credit for this section goes to matthiasplappert https://github.com/matthiasplappert/keras-rl ----------
        nb_action = model.output._keras_shape[-1]
        layer = model.layers[-2]
        y = Dense(nb_action + 1, activation='linear')(layer.output)
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                             output_shape=(nb_action,))(y)
        model = Model(input=model.input, output=outputlayer)
        # - - - - - - - - - - - - - - - - - ----------------------------------------------------------------------

        self.beh_model = models.clone_model(model)
        self.tar_model = models.clone_model(model)

    def check_env_compatibility(self, action_size, state_dim):
        # Make sure agent is compatible for this env
        # Check to make sure last layer's number of nodes is equal to number of actions
        if self.beh_model.layers[-1].output_shape[1] is not action_size:
            raise ValueError(
                'Agent is not compatible with this env, number of output nodes not equal to num of actions')
        if self.beh_model.layers[0].input_shape[1] is not state_dim:
            raise ValueError(
                'Agent is not compatible with this env, number of input nodes not equal to state dim')

    def act(self, state, state_dim):  # s.reshape(1, self.stateCnt)
        output = self.beh_model.predict(state.reshape(1, state_dim)).flatten()
        return self.currently_used_policy.return_action(output, ReturnActionType.Q_VALS)

    def remember(self, state, action, reward, next_state):
        self.memory.add((state, action, reward, next_state))

    def update_params(self, state_dim, action_size):
        minibatch = self.memory.sample(self.batch_size)

        blank_state = np.zeros(state_dim)

        states = np.array([m[0] for m in minibatch])
        states_ = np.array([(blank_state if m[3] is None else m[3]) for m in minibatch])

        beh_predictions = self.beh_model.predict(states)
        tar_predictions = self.tar_model.predict(states_)

        x = np.zeros((self.batch_size, state_dim))
        y = np.zeros((self.batch_size, action_size))
        index = 0
        for state, action, reward, next_state in minibatch:
            target = beh_predictions[index]
            if next_state is None:
                target[action] = reward
            else:
                if not self.double_dqn:
                    target[action] = reward + self.gamma * np.amax(tar_predictions[index])
                else:
                    target[action] = reward + self.gamma * tar_predictions[index][np.argmax(beh_predictions[index])]
            x[index] = state
            y[index] = target
            index += 1
        self.beh_model.train_on_batch(x, y)

    def summary(self):
        text = super().summary()
        text += 'Double option used: {}\n'.format(self.double_dqn)
        text += 'Dueling option used: {}\n'.format(self.dueling_dqn)
        text += 'Memory replay batch size: {}\n'.format(self.batch_size)
        text += 'Memory length: {}\n'.format(self.memory.storage.maxlen)
        return text
