'''
Outline:
- has a method (run) that contains a loop for the agent to run through the env
'''

class Runner:

    def run(self, agent, mode, env, nb_steps, visualize):
        assert mode == 'train' or mode == 'test'

        steps = 0
        for step in range(nb_steps):
            # Act

            if visualize:
                env.render()

            # Step
            # Check if terminal state
            # Remember
            # update
            # policy (epsilon) update
            # target models update
            # check if done
            # set state to next state
