from runner import Runner
from keras import models
from util import RunType
import numpy as np

'''
CURRENT WORK:
- Figure out how to use the **args and **kargs or whatever for making it so i can use default args in child classes
without needing to put their def values in the child class init if i want it to be universal for all child classes


Outline:
- this is the base abstract class, the algorithms will be in seperate files
- also contains methods like act, observe, remember ect.

Notes:
- act() assumes that all agents' models only use the state as input
- for train i use num of steps, for test num of epsiodes, not sure if this is best
- also for train and test might wanna add some other arguments such as steps for observing env (def is 1) ect.
- figure out how to do soft updates for check_update_target_model
- remember that in the training method of each specific alg if it uses replay to make sure the mem is full first *****
- rn the memory stuff is in the base agent class but i might wanna make it per alg
- im pretty sure i only need to compile the beh model cuz the tar model only ever uses predict aka its params are never optimized

- REMEMBER: to set uses_replay to false or true in each specific agent init, ALSO set agent_type, Make sure all agents call and add to summary()
- IMPORTANT: remember not to measure any metrics in these, only in callbacks, cuz of random memory fill
'''
from policy import EpsilonGreedyPolicy


class Agent:
    def __init__(self, model, optimizer, policy, gamma=0.95, target_model_update_policy='hard', target_model_hard_policy_wait=1000, target_model_soft_policy_constant=0.9):
        if target_model_hard_policy_wait < 1:
            raise ValueError('`target_model_hard_policy_wait` is < 1.')
        if gamma < 0 or gamma > 1:
            raise ValueError('`gamma` is < 0 or > 1.')
        if target_model_update_policy != 'hard' and target_model_update_policy != 'soft':
            raise ValueError('`target_model_update_policy` is not hard nor soft')
        self.beh_model = model
        self.tar_model = models.clone_model(model)
        self.optimizer = optimizer
        self.policy = policy
        self.currently_used_policy = None  # this is for temp setting it to like random for mem filling or greedy for testing
        self.gamma = gamma
        self.target_model_update_policy = target_model_update_policy
        self.target_model_hard_policy_wait = target_model_hard_policy_wait
        self.target_model_soft_policy_constant = target_model_soft_policy_constant
        # self.memory = Memory(max_memory_length) // Moved to agent specific
        self.uses_replay = None

        # Extra benchmarking info:
        self.agent_type = None
        self.number_of_trained_steps = 0  # Total number of steps this agent has ever been trained for
        self.training_sess_nb_steps_ep_max = None  # This is what the ep run ceiling was set to while training

    # Abstract method
    def act(self, state):
        raise NotImplementedError()

    # This is still in the base class so i can have it in the runner, use pass if its not needed in a specific agent
    def remember(self, state, action, reward, next_state):
        raise NotImplementedError()

    # Abstract method
    def update_params(self, step):
        raise NotImplementedError()

    def update_implicit_policy(self, step):
        self.policy.update(step)

    def check_update_target_model(self, step):
        if self.target_model_update_policy == 'hard':
            if step % self.target_model_hard_policy_wait == 0:
                self.tar_model.set_weights(self.beh_model.get_weights())
        else:
            tau = self.target_model_soft_policy_constant
            self.tar_model.set_weights(tau * np.asarray(self.beh_model.get_weights()) + (1 - tau) * np.asarray(self.tar_model.get_weights()))

    def train(self, env, nb_steps, nb_steps_ep_max=None, print_rew_cb=None, print_eps_cb=None, visualize=False, allow_printing=True):
        if type(self.policy) is EpsilonGreedyPolicy and print_eps_cb is not None:
            self.policy.set_cb(print_eps_cb)
        self.currently_used_policy = self.policy
        Runner(RunType.TRAIN, self, env, nb_steps, nb_steps_ep_max, print_rew_cb, print_eps_cb, None, visualize, allow_printing).run()

    # If they want benchmarking send in the file_name otherwise send in None (def.)
    def test(self, env, nb_steps, nb_steps_ep_max=None, print_rew_cb=None, benchmark_file_name=None, visualize=False, allow_printing=True):
        # The policy auto switches to greedy when testing so no epsilon cb even allowed
        Runner(RunType.TEST, self, env, nb_steps, nb_steps_ep_max, print_rew_cb, None, benchmark_file_name, visualize, allow_printing).run()

    def summary(self):
        text = "Agent Details:\n"
        assert self.agent_type is not None, 'Need to set `agent_type`'
        text += "Agent type: {}\n".format(self.agent_type.value)
        text += "Model details: MODEL SUMMARIZER NOT IMPLEMENTED YET\n"
        # agent.beh_model.summary(print_fn=lambda x: f.write(x + '\n'))
        text += "Policy: {}\n".format(self.policy.summary())
        text += "Gamma: {}\n".format(self.gamma)
        text += "Target model update type: {}\n".format(self.target_model_update_policy.title())
        if self.target_model_update_policy is 'hard':
            text += "Target model update wait: {}\n".format(self.target_model_hard_policy_wait)
        else:
            text += "Target model update constant: {}\n".format(self.target_model_soft_policy_constant)
        return text
