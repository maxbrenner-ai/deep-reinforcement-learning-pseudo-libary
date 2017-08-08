from runner import Runner
from memory import Memory

'''
Outline:
- this is the base abstract class, the algorithms will be in seperate files
- also contains methods like act, observe, remember ect.

Notes:
- act() assumes that all agents' models only use the state as input
'''


class Agent:

    def __init__(self, model, optimizer, policy, max_memory_length=1000):
        self.beh_model = model
        # !!!! Find out how to deep copy a model for the target model
        self.optimizer = optimizer
        self.policy = policy
        self.memory = Memory(max_memory_length)

    def act(self, state):
        output = self.beh_model.predict(state)  # assumes all models only use state as input
        self.policy.return_action(output)

    def remember(self, state, action, reward, next_state):
        self.memory.add((state, action, reward, next_state))

    # def update_params(self):

    def update_implicit_policy(self):
        self.policy.update()

    # def check_update_target_model(self):

    def train(self, env, nb_steps, visualize):
        runner = Runner(self, 'train', env, nb_steps, visualize)

    def test(self, env, nb_steps, visualize):
        runner = Runner(self, 'test', env, nb_steps, visualize)
