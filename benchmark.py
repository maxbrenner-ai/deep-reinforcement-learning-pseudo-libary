'''
This class is for saving the details of an agent in a readable format to an external file, along with the rewards for
each episode, the avg reward per episode and the total reward for the test run

Things to maybe add:
- Allow multiple tests on one benchmark file for an agent
- Once i do some more work with the agents, like add dueling DQN revisit the model summary output and figure that out
'''
from prettytable import PrettyTable as table
from util import AgentType


class Benchmark:
    def __init__(self, file_path, episode_iteration=1):
        if episode_iteration < 1:
            raise ValueError("`episode_iteration` should not be less than 1")
        self.file_path = file_path
        self.episode_iteration = episode_iteration

        self.episode_reward_logs = []
        self.total_reward = 0
        self.total_ep_reward = 0
        self.current_avg = 0

    def update(self, reward):
        assert reward is not None
        self.total_reward += reward
        self.total_ep_reward += reward

    def update_end_of_ep(self, episode):
        self.current_avg = self.current_avg + ((self.total_ep_reward - self.current_avg) / episode)
        self.episode_reward_logs.append([episode, self.total_ep_reward])
        self.total_ep_reward = 0

    def output_to_file(self, end_of_run, epsiode, agent_summary, runner_summary):
        if epsiode % self.episode_iteration == 0 or end_of_run:
            t = table(['Episode', 'Total Episode Reward'])
            for row in self.episode_reward_logs:
                t.add_row(row)

            f = open(self.file_path, 'w')

            # First print agent details
            text = agent_summary + '\n'
            text += runner_summary
            text += t.get_string() + '\n'
            text += "Total run reward: {}\n".format(self.total_reward)
            text += "Avg episode reward: {}\n".format(self.current_avg)

            f.write(text)

            f.close()

