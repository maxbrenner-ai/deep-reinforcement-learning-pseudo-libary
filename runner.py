'''
Outline:
- has a method (run) that contains a loop for the agent to run through the env

Working on:
- might wanna change it to a static method or put it in the agent base class

Notes:
- remember if i need to check in other classes if this is a termination state check next_state not state
- idk if the policy should always be full GREEDY when TESTING?
'''


class Runner:

    def run_train(self, agent, env, nb_steps, nb_steps_ep_max, visualize):

        current_ep_step = 0
        # initial state
        state = env.reset()
        for step in range(nb_steps):
            current_ep_step += 1

            action = agent.act(state)

            if visualize:
                env.render()

            next_state, reward, done, _ = env.step(action)

            if done:
                next_state = None

            agent.remember(state, action, reward, next_state)

            agent.update_params()

            agent.update_implicit_policy()

            agent.check_update_target_model()

            # If next_state is terminal or the current ep steps is equal to the number of steps max for an episode
            # then reset, otherwise continue
            if done or current_ep_step == nb_steps_ep_max:
                current_ep_step = 0
                state = env.reset()
            else:
                state = next_state

    # Implement this
    def run_test(self, agent, env, nb_episodes, nb_steps_ep_max, visualize):
        i = 4