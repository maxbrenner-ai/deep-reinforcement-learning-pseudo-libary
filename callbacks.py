'''
Outline:
- list of callbacks
- not sure about this one and in comparison to metrics and benchmarks

List of Callbacks:
- Print reward: total after every episode, total and avg per episode at end 

- Print epsilon
    - Every step or every ep, or every some number of iterations
    
NOTES:
IMPORTANT: Disable all PrintCallback functions from being called or doing anything if run_type is RAND_FILL
- I still need that check cuz even tho i make a new instance of runner for randfill and send in no cbs, the policy
has a cb possibly and it wont know what the run type is
- Duplicates are not allowed and have been fixed

- Unfortunately when i make a new callback i will have to send its reference directly to where it needs to go, and
call update directly
'''

from utils import RunType

# Manager class for managing the signals (flags) and printing or updating the print calbacks based off of the flgas
class PrintCallbacksManager:
    def __init__(self, callbacks, run_type):
        self.callbacks = callbacks
        self.run_type = run_type
        self.set_run_type_of_cbs(run_type)

    def set_run_type_of_cbs(self, run_type):
        self.run_type = run_type
        for cb in self.callbacks:
            cb.current_run_type = run_type

    def refresh_cbs(self):
        for cb in self.callbacks:
            cb.refresh()

    def access_callbacks(self, allow_printing, end_of_step=False, step=None, end_of_episode=False, episode=None, end_of_run=False):
        # Disable all callbacks from updaing or printing when randomly filling
        if self.run_type is RunType.RAND_FILL:
            return
        for cb in self.callbacks:
            if end_of_step:
                assert step is not None
                if allow_printing:
                    cb.print(True, step, False, None, False)
                cb.update_on_flag(True, step, False, None, False)
            if end_of_episode:
                assert episode is not None
                if allow_printing:
                    cb.print(False, None, True, episode, False)
                cb.update_on_flag(False, None, True, episode, False)
            if end_of_run:
                if allow_printing:
                    cb.print(False, None, False, None, True)
                cb.update_on_flag(False, None, False, None, True)


# Abstract class
class PrintCallback:
    def __init__(self):
        self.current_run_type = None

    # This method makes it so the user does not have to make a new callback to use the same one with the same params
    # for say testing after training, this will be called auto in the runner after run is complete
    def refresh(self):
        raise NotImplementedError

    # Observer update method
    def update(self, value):
        raise NotImplementedError

    def print(self, end_of_step, step, end_of_episode, episode, end_of_run):
        raise NotImplementedError

    # For updating something based on end of episode or step or run
    def update_on_flag(self, end_of_step, step, end_of_episode, episode, end_of_run):
        raise NotImplementedError


class PrintReward(PrintCallback):
    def __init__(self):
        super().__init__()
        self.refresh()

    def refresh(self):
        self.total_reward = 0
        self.total_ep_reward = 0
        self.current_avg = 0

    def update(self, reward):
        assert self.current_run_type is not None
        if self.current_run_type is RunType.RAND_FILL:
            return
        assert reward is not None
        self.total_reward += reward
        self.total_ep_reward += reward

    def print(self, end_of_step, step, end_of_episode, episode, end_of_run):
        if end_of_episode:
            print("Total Episode Reward:", self.total_ep_reward)
        if end_of_run:
            # Know that the avg will not be equal to the total/num of ep in the end cuz it will prolly end in the middle of the last episode
            print("Total Run Reward:", self.total_reward)
            print("Avg Episode Reward:", self.current_avg)

    def update_on_flag(self, end_of_step, step, end_of_episode, episode, end_of_run):
        if end_of_episode:
            # Running Episode avg
            self.current_avg = self.current_avg + ((self.total_ep_reward - self.current_avg) / episode)
            self.total_ep_reward = 0


class PrintEpsilon(PrintCallback):
    def __init__(self, episodic=True, iterations=None):
        super().__init__()
        if episodic is False and (iterations is None or iterations < 1):
            raise ValueError("`episodic` is False which means `iterations` should be a number greater than 0")
        self.episodic = episodic
        self.iterations = iterations
        self.epsilon = None

    def refresh(self):
        pass

    def update(self, epsilon):
        assert self.current_run_type is not None
        if self.current_run_type is RunType.RAND_FILL:
            return
        assert epsilon is not None
        self.epsilon = epsilon

    def print(self, end_of_step, step, end_of_episode, episode, end_of_run):
        if self.iterations is not None:
            if end_of_step and (step+1) % self.iterations == 0:
                assert self.epsilon is not None
                print("Epsilon:", self.epsilon)
        if end_of_episode and self.episodic:
            assert self.epsilon is not None
            print("Epsilon:", self.epsilon)

    def update_on_flag(self, end_of_step, step, end_of_episode, episode, end_of_run):
        pass
