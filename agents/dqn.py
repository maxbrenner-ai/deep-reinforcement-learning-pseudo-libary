from agents.base import Agent
from policy import ReturnActionType
from memory import DequeMemory, SumTreeMemory
from util import AgentType
from keras import models
from keras.models import Model
from keras.layers import *
import numpy as np
import tensorflow as tf

'''
- Might need to add TD-error clipping to [-1, 1]
- Not sure if new class for PER just for holding values is a good idea, but they are only used if PER is in effect so
i guess its fine
- IDk what linearly annealing a start value of b to 1 means??????
'''
class PER:
    def __init__(self, priority_importance=0.5, edge_add=0.1, initial_anneal=0.5):
        self.a = priority_importance
        self.e = edge_add
        self.b = initial_anneal


class DQN(Agent):
    def __init__(self, double_dqn=True, PER=PER(), dueling_dqn=False, add_dueling_streams=False, batch_size=32, max_memory_length=1000, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)

        if max_memory_length < 1:
            raise ValueError('`max_memory_length` is < 1.')
        if batch_size < 1 or batch_size > max_memory_length:
            raise ValueError('`batch_size` is < 1 or > `max_memory_length`')
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.PER = PER
        if self.PER is None:
            self.memory = DequeMemory(max_memory_length)
        else:
            self.memory = SumTreeMemory(max_memory_length, self.PER.a, self.PER.e)

        # Dueling set-up ------
        self.dueling_dqn = dueling_dqn
        # So the user can send in an arch that has the streams already made, if so just create the TF graph
        self.add_dueling_streams = add_dueling_streams
        if dueling_dqn is False and add_dueling_streams is True:
            raise ValueError("`add_dueling_streams` cannot be true if `dueling_dqn` is false")
        if self.dueling_dqn:
            self.sess = tf.Session()
            K.set_session(self.sess)
            # This says that I will handle the init i.e. will call global_variables_initializers() in sess.run
            K.manual_variable_initialization(True)
            self.create_placeholders()
            if self.add_dueling_streams:
                self.beh_model = self.construct_dueling_streams(self.beh_model)
                # Reclone target model cuz the behavior model has been changed
                self.tar_model = models.clone_model(self.beh_model)
            self.create_beh_graph()
            self.create_tar_graph()
            self.sess.run(tf.global_variables_initializer())
        # ---------------------

        # Don't want to compile if dueling is true cuz TF is used for compiling then
        if not dueling_dqn:
            self.beh_model.compile(loss='mse', optimizer=self.optimizer)

        self.uses_replay = True
        self.agent_type = AgentType.DQN

    # DUELING ----------------------------------------------------------------------------------------------------------

    def create_placeholders(self):
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='state')  # state
        self.targets = tf.placeholder(tf.float32, shape=(None, self.action_size), name='target')  # q_vals

    # This will take the second to last layer duplicate it, on one stream put the advantage part and the other the state
    def construct_dueling_streams(self, model):
        # Check to make sure more than one layer (idk why there would be one layer anyway)
        if len(model.layers) < 1:
            raise ValueError('Model needs more than one layers for dueling to work')

        model.layers.pop()
        adv_stream = Dense(self.action_size, activation='linear')(model.layers[-1].output)
        state_stream = Dense(1, activation='linear')(model.layers[-1].output)
        model = Model(input=model.input, outputs=[adv_stream, state_stream])

        return model

    def create_beh_graph(self):
        a_out, s_out = self.beh_model(self.states)
        self.beh_output = s_out + tf.subtract(a_out, tf.reduce_mean(a_out, reduction_indices=1, keep_dims=True))
        # MSE (target - output)^2, reduce_mean OR reduce_sum IDK!
        loss = tf.reduce_mean(tf.square(tf.subtract(self.targets, self.beh_output)))

        optimizer = self.optimizer
        self.minimize = optimizer.minimize(loss)

    def create_tar_graph(self):
        a_out, s_out = self.tar_model(self.states)
        self.tar_output = s_out + tf.subtract(a_out, tf.reduce_mean(a_out, reduction_indices=1, keep_dims=True))

    # ------------------------------------------------------------------------------------------------------------------

    def check_env_compatibility(self, action_size, state_dim):
        # Make sure agent is compatible for this env
        # Check to make sure last layer's number of nodes is equal to number of actions
        if self.action_size is not action_size:
            raise ValueError(
                'Agent is not compatible with this env, number of output nodes not equal to num of actions')
        if self.state_dim is not state_dim:
            raise ValueError(
                'Agent is not compatible with this env, number of input nodes not equal to state dim')

    def act(self, state):  # s.reshape(1, self.stateCnt)
        output = self.predict(state.reshape(1, self.state_dim), target=False).flatten()
        return self.currently_used_policy.return_action(output, ReturnActionType.Q_VALS)

    def remember(self, state, action, reward, next_state, get_error):
        sample = (state, action, reward, next_state)
        # If get_error is true than get the TD error and send it in
        error =  None
        if get_error and type(self.memory) is SumTreeMemory:
            (_,  _, errors) = self.return_inputs_targets_errors([sample])
            error = errors[0] # Will be a list so just get first element
        self.memory.add(sample, error=error)

    def predict(self, states, target):
        if self.dueling_dqn:
            if not target:
                return self.sess.run(self.beh_output, {self.states: states})
            else:
                return self.sess.run(self.tar_output, {self.states: states})
        else:
            if not target:
                return self.beh_model.predict(states)
            else:
                return self.tar_model.predict(states)

    # Private method
    def return_inputs_targets_errors(self, minibatch):
        # We need this just as a placeholder for an empty state
        blank_state = np.zeros(self.state_dim)

        states = np.array([m[0] for m in minibatch])
        next_states = np.array([blank_state if m[3] is None else m[3] for m in minibatch])

        beh_predictions = self.predict(states, False)
        tar_predictions = self.predict(next_states, True)

        x = np.zeros((self.batch_size, self.state_dim))
        y = np.zeros((self.batch_size, self.action_size))
        td_errors = np.zeros(self.batch_size)
        index = 0

        for state, action, reward, next_state in minibatch:
            target = beh_predictions[index]
            old_value = target[action]  # The original
            if next_state is None:
                target[action] = reward
            else:
                if not self.double_dqn:
                    target[action] = reward + self.gamma * np.amax(tar_predictions[index])
                else:
                    target[action] = reward + self.gamma * tar_predictions[index][np.argmax(beh_predictions[index])]

            x[index] = state
            y[index] = target
            td_errors[index] = abs(target[action] - old_value)  # Gets the |TD Error|
            index += 1

        return (x, y, td_errors)

    def update_params(self):
        minibatch_data, minibatch_indices = self.memory.sample_batch(self.batch_size)
        (x, y, errors) = self.return_inputs_targets_errors(minibatch_data)
        # Update the priorities
        self.memory.update(minibatch_indices, errors)
        if not self.dueling_dqn:
            self.beh_model.train_on_batch(x, y)
        else:
            feed_dict = {self.states: x, self.targets: y}
            self.sess.run(self.minimize, feed_dict)


    def summary(self):
        text = super().summary()
        text += 'Double option used: {}\n'.format(self.double_dqn)
        text += 'Dueling option used: {}\n'.format(self.dueling_dqn)
        text += 'Added dueling streams: {}\n'.format(self.add_dueling_streams)
        text += 'Memory replay batch size: {}\n'.format(self.batch_size)
        text += 'Memory length: {}\n'.format(self.memory.storage.maxlen)
        return text
