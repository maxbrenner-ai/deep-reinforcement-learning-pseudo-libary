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

- IMPORTANT: the instance var run_type is CONSTANT, it never changes and should never change from its initial state in
the runner
- IMPORTANT: Make sure i dont send any callbacks when rand filling
- IMPORTANT: Make sure i do auto replace the policy of the agent when testing with greedy, cuz its right but also because
if i dont, then the runner will use the same instance of the policy (eps greedy) and the eps will be phucked up and
a cb might still be in there
'''
import numpy as np
from policy import RandomPolicy, GreedyPolicy
from callbacks import PrintCallbacksManager
from utils import RunType


class Runner:
    def __init__(self, run_type, agent, env, nb_steps, nb_steps_ep_max=None, print_rew_cb=None, print_eps_cb=None, visualize=False, allow_printing=True):
        self.run_type = run_type
        self.agent = agent
        self.env = env
        self.nb_steps = nb_steps
        self.nb_steps_ep_max = nb_steps_ep_max
        self.visualize = visualize
        self.allow_printing = allow_printing

        # Make a callbacksmanager
        callbacks = []
        if print_rew_cb is not None:
            callbacks.append(print_rew_cb)
        if print_eps_cb is not None:
            callbacks.append(print_eps_cb)
        self.cbmanager = PrintCallbacksManager(callbacks, self.run_type)
        self.print_rew_cb = print_rew_cb

        assert agent.uses_replay is not None, "`uses_replay` is still `None`, need to set it."

    def run(self):
        if self.run_type is RunType.TRAIN:
            assert self.agent.currently_used_policy is self.agent.policy, "While training the current policy should be the init one"

        # If it needs to fill the mem and its not already doing that then fill it
        if self.agent.uses_replay is True and self.run_type is RunType.TRAIN:
            self.cbmanager.set_run_type_of_cbs(RunType.RAND_FILL)
            mem_filler = Runner(RunType.RAND_FILL, self.agent, self.env, None, self.nb_steps_ep_max, None, None, False, self.allow_printing)
            mem_filler.run()
            self.cbmanager.set_run_type_of_cbs(self.run_type)

        self.print_flag()

        # Now temp replace the pol if rand_fill or test
        self.temp_replace_policy()

        state_size = self.env.observation_space.shape[0]
        current_total_step = 0
        current_episode = 1  # MUST START AT 1
        current_ep_step, state = self.reset_eipsode(state_size)

        if self.allow_printing and self.run_type is not RunType.RAND_FILL:
            print("Episode", current_episode, "...")

        while self.check_loop(current_total_step):

            # Debug ----
            if self.agent.uses_replay is True and self.run_type is RunType.TRAIN:
                assert self.agent.memory.is_full(), "Replay agent's memory is NOT full while training."
            # ----------

            action = self.agent.act(state)

            if self.visualize:
                self.env.render()

            next_state, reward, done, _ = self.env.step(action)
            if self.print_rew_cb is not None:
                self.print_rew_cb.update(reward)

            next_state = np.reshape(next_state, [1, state_size])

            if done:
                next_state = None

            self.remember(state, action, reward, next_state)
            # print(len(self.agent.memory.storage))

            self.update_models_and_policy(current_total_step)

            if done or current_ep_step == self.nb_steps_ep_max:
                # End of episode
                self.cbmanager.access_callbacks(self.allow_printing, end_of_episode=True, episode=current_episode)
                current_ep_step, state = self.reset_eipsode(state_size)
                current_episode += 1

                if self.allow_printing and self.run_type is not RunType.RAND_FILL:
                    print("Episode", current_episode, "...")
            else:
                state = next_state
                current_ep_step += 1

            # End of step
            self.cbmanager.access_callbacks(self.allow_printing, end_of_step=True, step=current_total_step)
            current_total_step += 1

        # End of run
        self.cbmanager.access_callbacks(self.allow_printing, end_of_run=True)

        if self.allow_printing:
            print("...Done")
        # At very end reset the policy
        self.reset_policy()
        # Refresh cbs in case they are used again
        self.cbmanager.refresh_cbs()

    def print_flag(self):
        if self.allow_printing:
            if self.run_type is RunType.RAND_FILL:
                print("Randomly filling agent memory...")
            elif self.run_type is RunType.TRAIN:
                print("Training agent...")
            elif self.run_type is RunType.TEST:
                print("Testing agent...")

    def temp_replace_policy(self):
        if self.run_type is RunType.RAND_FILL:
            self.agent.currently_used_policy = RandomPolicy()
        if self.run_type is RunType.TEST:
            self.agent.currently_used_policy = GreedyPolicy()

    def reset_eipsode(self, state_size):
        return 0, np.reshape(self.env.reset(), [1, state_size])

    def check_loop(self, current_total_step):
        if self.run_type is RunType.RAND_FILL:
            if self.agent.memory.is_full() is False:
                return True
            else:
                return False
        if self.run_type is RunType.TRAIN or self.run_type is RunType.TEST:
            if current_total_step == self.nb_steps:
                return False
            else:
                return True

    def remember(self, state, action, reward, next_state):
        if self.run_type is RunType.TEST:
            return
        self.agent.remember(state, action, reward, next_state)

    def update_models_and_policy(self, current_total_step):
        if self.run_type is RunType.RAND_FILL or self.run_type is RunType.TEST:
            return
        # CURRENTLY: the update_implicit_policy is in the update_params method
        # if current_total_step % 10 == 0:
        self.agent.update_params(current_total_step)
        self.agent.update_implicit_policy(current_total_step)

        # this one can update on any step probs
        self.agent.check_update_target_model(current_total_step)

    def reset_policy(self):
        self.agent.currently_used_policy = self.agent.policy
