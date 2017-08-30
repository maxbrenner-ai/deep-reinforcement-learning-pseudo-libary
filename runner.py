'''
Outline:
- has a method (run) that contains a loop for the agent to run through the env

Notes:
- remember if i need to check in other classes if this is a termination state check next_state not state
- idk if the policy should always be full GREEDY when TESTING? For now it is
- idk if update params should be every step, or some number of steps 
- I think that for agents that use memory/replay a random policy should be used to fill the memory and this should not
count against the agent
- Currently the policy only updates when the params update

- SOMETHING TO CONSIDER: instead of making a runner instance every call to train or test, maybe just make one in the
init of the agent????
'''
import numpy as np
from policy import RandomPolicy, GreedyPolicy, ReturnActionType
from enum import Enum

'''












Current Work: HERE IS HOW I AM GONNA DO IT:
    - Every piece of code in the runner will have its own method and i will send a flag in which is
    random_fill, train or test and add any ifs in each method 
    - This way i can have ONE run method and i just send in the appropriate flag
    - And cuz the structure is slightly dif (the loops) just do a while loop and have it call a method, once again
    send in the flag













'''
class RunType(Enum):
    RAND_FILL = 1
    TRAIN = 2
    TEST = 3

class Runner:
    def __init__(self, run_type, agent, env, nb_steps=None, nb_steps_ep_max=None, visualize=False):
        self.run_type = run_type
        self.agent = agent
        self.env = env
        self.nb_steps = nb_steps
        self.nb_steps_ep_max = nb_steps_ep_max
        self.visualize = visualize

        assert agent.uses_replay is not None, "`uses_replay` is still `None`, need to set it."

    # def run(self):
    #     if self.run_type is RunType.TRAIN:
    #         assert self.agent.currently_used_policy is self.agent.policy, "While training the current policy should be the init one"
    #
    #     # If it needs to fill the mem and its not already doing that then fill it
    #     if self.agent.uses_replay is True and self.run_type is not RunType.RAND_FILL:
    #         mem_filler = Runner(RunType.RAND_FILL, self.agent, self.env, None, self.nb_steps_ep_max, False)
    #         mem_filler.run()
    #
    #     # Now temp replace the pol if rand_fill or test
    #     self.temp_replace_policy()
    #
    #
    #
    #
    #     # At very end reset the policy
    #     self.reset_policy()
    #
    # def temp_replace_policy(self):
    #     if self.run_type is RunType.RAND_FILL:
    #         self.agent.currently_used_policy = RandomPolicy()
    #     if self.run_type is RunType.TEST:
    #         self.agent.currently_used_policy = GreedyPolicy()
    #
    #
    # def reset_policy(self):
    #     self.agent.currently_used_policy = self.agent.policy

    # Here is where i randomly fill the memory up for the agent, and have it not count against the stats
    def fill_memory(self, agent, env, nb_steps_ep_max):
        print("Filling Memory...")

        # !!!! Start by temp replacing the pol with a random one
        agent.currently_used_policy = RandomPolicy()

        state_size = env.observation_space.shape[0]

        current_ep_step = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while agent.memory.is_full is False:

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if done:
                next_state = None

            agent.remember(state, action, reward, next_state)

            if done or current_ep_step == nb_steps_ep_max:
                current_ep_step = 0
                state = env.reset()
                state = np.reshape(state, [1, state_size])

            else:
                state = next_state
                current_ep_step += 1

        # set policy back
        agent.currently_used_policy = agent.policy

    def run_train(self, agent, env, nb_steps, nb_steps_ep_max=None, visualize=False):
        # Make sure current policy is = to policy
        assert agent.currently_used_policy is agent.policy, "While training the current policy should be the init one"

        # First use a random policy to fill the memory
        assert agent.uses_replay is not None, "`uses_replay` is still `None`, need to set it."
        if agent.uses_replay is True:
            self.fill_memory(agent, env, nb_steps_ep_max)

        state_size = env.observation_space.shape[0]

        current_ep_step = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for step in range(nb_steps):

            # Debug ----
            if agent.uses_replay is True:
                assert agent.memory.is_full(), "Replay agent's memory is NOT full while training."
            # ----------

            action = agent.act(state)

            if visualize:
                env.render()

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if done:
                next_state = None

            agent.remember(state, action, reward, next_state)

            # CURRENTLY: the update_implicit_policy is in the update_params method
            # if step % 500 == 0:
            agent.update_params(step)
            agent.update_implicit_policy(step)

            # this one can update on any step probs
            agent.check_update_target_model(step)

            # If next_state is terminal or the current ep steps is equal to the number of steps max for an episode
            # then reset, otherwise continue
            if done or current_ep_step == nb_steps_ep_max:

                # print(current_ep_step)

                current_ep_step = 0
                state = env.reset()
                state = np.reshape(state, [1, state_size])

            else:
                state = next_state
                current_ep_step += 1

    # Implement this
    def run_test(self, agent, env, nb_episodes, nb_steps_ep_max=None, visualize=False):
        agent.currently_used_policy = GreedyPolicy()

        state_size = env.observation_space.shape[0]

        for episode in nb_episodes:
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            current_ep_step = 0

            while current_ep_step is not nb_steps_ep_max:
                action = agent.act(state)

                if visualize:
                    env.render()

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                if done:
                    next_state = None
                    break

                state = next_state
                current_ep_step += 1

        agent.currently_used_policy = agent.policy
