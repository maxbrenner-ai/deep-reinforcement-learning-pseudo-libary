from agents.base import Agent
from policy import ReturnActionType
from memory import DequeMemory, SumTreeMemory
from util import AgentType
from keras import models
from keras.models import Model
from keras.layers import *
import numpy as np
import tensorflow as tf
from math import exp


class PER:
    # Can set growth to 0
    def __init__(self, priority_importance=0.5, initial_anneal=0.5, anneal_growth_rate=0.00008):
        self.a = priority_importance  # Constant
        self.e = 0.1  # Constant
        self.b = initial_anneal  # Can be Dyanmic
        self.initial_anneal = initial_anneal  # Constant
        self.growth = anneal_growth_rate  # Constant

    # Linear anneal
    def update_b(self):
        self.b = self.b + (1.0 - self.initial_anneal) * self.growth
        self.b = min(self.b, 1.0)


class DQN(Agent):
    def __init__(self, double_dqn=True, PER=None, dueling_dqn=False, add_dueling_streams=False, huber_loss=False,
                 batch_size=32, max_memory_length=1000, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)

        if max_memory_length < 1:
            raise ValueError('`max_memory_length` is < 1.')
        if batch_size < 1 or batch_size > max_memory_length:
            raise ValueError('`batch_size` is < 1 or > `max_memory_length`')
        self.huber_loss = huber_loss
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.PER = PER
        if self.PER is None:
            self.memory = DequeMemory(max_memory_length)
            self.using_PER = False
        else:
            self.memory = SumTreeMemory(max_memory_length, self.PER.a, self.PER.e)
            self.using_PER = True

        self.dueling_dqn = dueling_dqn
        # So the user can send in an arch that has the streams already made, if so just create the TF graph
        self.add_dueling_streams = add_dueling_streams
        if dueling_dqn is False and add_dueling_streams is True:
            raise ValueError("`add_dueling_streams` cannot be true if `dueling_dqn` is false")

        # Compiling -------------------
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
        # -----------------------------

        self.max_memory_length = max_memory_length
        self.uses_replay = True
        self.agent_type = AgentType.DQN

    # COMPILING --------------------------------------------------------------------------------------------------------

    def create_placeholders(self):
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='states')  # state
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.targets = tf.placeholder(tf.float32, shape=(None,), name='targets')  # q_vals
        self.IS_weights = tf.placeholder(tf.float32, shape=(None,), name='IS_weights')  # For PER

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
        if self.dueling_dqn:
            a_out, s_out = self.beh_model(self.states)
            self.beh_output = s_out + tf.subtract(a_out, tf.reduce_mean(a_out, axis=1, keep_dims=True))
        else:
            self.beh_output = self.beh_model(self.states)

        # Index outputs by actions and convert shape to a vector
        outputs_vec = tf.reduce_sum(tf.multiply(tf.one_hot(self.actions, self.action_size), self.beh_output), axis=1)
        errors = tf.subtract(self.targets, outputs_vec)

        if self.huber_loss:
            td_errors = tf.abs(errors)
            quadratic_part = tf.minimum(td_errors, 1.0)
            first = 0.5 * tf.square(quadratic_part)
            second = 1.0 * tf.subtract(td_errors, quadratic_part)
            loss = tf.add(first, second)
        else:
            # MSE
            loss = tf.square(errors)
        # Need to multiply the losses by the IS weights if using PER
        if self.using_PER:
            loss = tf.multiply(self.IS_weights, loss)

        loss = tf.reduce_mean(loss)

        optimizer = self.optimizer
        self.minimize = optimizer.minimize(loss)

    def create_tar_graph(self):
        if self.dueling_dqn:
            a_out, s_out = self.tar_model(self.states)
            self.tar_output = s_out + tf.subtract(a_out, tf.reduce_mean(a_out, axis=1, keep_dims=True))
        else:
            self.tar_output = self.tar_model(self.states)

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
            (_,  _, _, errors) = self.return_inputs_targets_errors([sample])
            error = errors[0] # Will be a list so just get first element
        self.memory.add(sample, error=error)

    def predict(self, states, target):
        if not target:
            return self.sess.run(self.beh_output, {self.states: states})
        else:
            return self.sess.run(self.tar_output, {self.states: states})

    # Private method
    def return_inputs_targets_errors(self, minibatch):
        # We need this just as a placeholder for an empty state
        blank_state = np.zeros(self.state_dim)

        states = np.array([m[0] for m in minibatch])
        next_states = np.array([blank_state if m[3] is None else m[3] for m in minibatch])

        beh_predictions = self.predict(states, False)
        tar_predictions = self.predict(next_states, True)

        x = np.zeros((self.batch_size, self.state_dim))
        y = np.zeros(self.batch_size)
        actions = np.zeros(self.batch_size)
        td_errors = np.zeros(self.batch_size)
        index = 0

        for state, action, reward, next_state in minibatch:
            # target = beh_predictions[index]
            old_value = beh_predictions[index][action]  # The original
            if next_state is None:
                target = reward
            else:
                if not self.double_dqn:
                    target = reward + self.gamma * np.amax(tar_predictions[index])
                else:
                    target = reward + self.gamma * tar_predictions[index][np.argmax(beh_predictions[index])]
            x[index] = state
            y[index] = target
            actions[index] = action
            td_errors[index] = abs(target - old_value)  # Gets the |TD Error|
            index += 1

        return (x, y, actions, td_errors)

    def update_params(self):
        minibatch_data, minibatch_indices = self.memory.sample_batch(self.batch_size)
        (x, y, actions, errors) = self.return_inputs_targets_errors(minibatch_data)
        # Update the priorities
        priors = self.memory.update(minibatch_indices, errors)

        # Make the IS Weights
        if self.using_PER:
            max_prior_weight = (self.max_memory_length * self.memory.current_min_priority) ** -self.PER.b
            is_weights = []
            for p in priors:
                new_weight = (self.max_memory_length * p) ** -self.PER.b
                new_weight = new_weight/max_prior_weight
                is_weights.append(new_weight)
            # Increase b
            self.PER.update_b()
        else:
            is_weights = [0]

        feed_dict = {self.states: x, self.targets: y, self.actions: actions, self.IS_weights: is_weights}
        self.sess.run(self.minimize, feed_dict)

    def summary(self):
        text = super().summary()
        text += "Double option used: {}\n".format(self.double_dqn)
        text += "Prioritized Experience Replay used: {}\n".format(self.using_PER)
        if self.using_PER:
            text += "Priority importance(a): {}\n".format(self.PER.a)
            text += "Initial annealing constant(b): {}\n".format(self.PER.initial_anneal)
            text += "Anneal growth rate: {}\n".format(self.PER.growth)
        text += "Dueling option used: {}\n".format(self.dueling_dqn)
        text += "Added dueling streams: {}\n".format(self.add_dueling_streams)
        text += "Memory replay batch size: {}\n".format(self.batch_size)
        text += "Memory length: {}\n".format(self.max_memory_length)
        return text
