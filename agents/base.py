from runner import Runner
from keras import models
from enum import Enum

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

- REMEMBER: to set uses_replay to false or true in each specific agent init
- IMPORTANT: remember not to measure any metrics in these, only in runner, cuz of random memory fill
'''


class TargetUpdateType(Enum):
    HARD = 1
    SOFT = 2


class Agent:
    def __init__(self, model, optimizer, policy, gamma=0.95, target_model_update_policy=TargetUpdateType.HARD, target_model_hard_policy_wait=1000):
        if target_model_hard_policy_wait < 1:
            raise ValueError('`target_model_hard_policy_wait` is < 1.')
        if gamma < 0 or gamma > 1:
            raise ValueError('`gamma` is < 0 or > 1.')
        self.beh_model = model
        self.tar_model = models.clone_model(model)
        self.optimizer = optimizer
        self.policy = policy
        self.currently_used_policy = policy  # this is for temp setting it to like random for mem filling or greedy for testing
        self.gamma = gamma
        self.target_model_update_policy = target_model_update_policy
        self.target_model_hard_policy_wait = target_model_hard_policy_wait
        # self.memory = Memory(max_memory_length) // Moved to agent specific
        self.uses_replay = None

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
        if self.target_model_update_policy.name == TargetUpdateType.HARD and step % self.target_model_hard_policy_wait == 0:
            self.tar_model.set_weights(self.beh_model.get_weights())
        # elif self.target_model_update_policy.name == TargetUpdateType.SOFT:
        #     IMPLEMENT THIS

    def train(self, env, nb_steps, nb_steps_ep_max=None, visualize=False):
        runner = Runner()
        runner.run_train(self, env, nb_steps, nb_steps_ep_max, visualize)

    def test(self, env, nb_episodes, nb_steps_ep_max=None, visualize=False):
        runner = Runner()
        runner.run_test(self, env, nb_episodes, nb_steps_ep_max, visualize)
