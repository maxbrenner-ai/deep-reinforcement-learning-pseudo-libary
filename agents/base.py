from runner import Runner
from memory import Memory

'''
Outline:
- this is the base abstract class, the algorithms will be in seperate files
- also contains methods like act, observe, remember ect.

Notes:
- act() assumes that all agents' models only use the state as input
- for train i use num of steps, for test num of epsiodes, not sure if this is best
- also for train and test might wanna add some other arguments such as steps for observing env (def is 1) ect.
'''


class Agent:

    def __init__(self, model, optimizer, policy, max_memory_length=1000):
        self.beh_model = model
        # !!!! Find out how to deep copy a model for the target model
        self.optimizer = optimizer
        self.policy = policy
        self.memory = Memory(max_memory_length)

    # Abstract method
    def act(self, state):
        raise NotImplementedError()
        # output = self.beh_model.predict(state)  # assumes all models only use state as input
        # self.policy.return_action(output, MODEEE) # Make sure to add mode if the policy is an EpsilonGreedy one

    def remember(self, state, action, reward, next_state):
        self.memory.add((state, action, reward, next_state))

    # Implement this
    def update_params(self):
        i = 4

    def update_implicit_policy(self):
        self.policy.update()

    # Implement this
    def check_update_target_model(self):
        i = 4

    def train(self, env, nb_steps, nb_steps_ep_max=None, visualize=False):
        runner = Runner()
        runner.run_train(self, env, nb_steps, nb_steps_ep_max, visualize)

    def test(self, env, nb_episodes, nb_steps_ep_max=None, visualize=False):
        runner = Runner()
        runner.run_test(self, env, nb_episodes, nb_steps_ep_max, visualize)
