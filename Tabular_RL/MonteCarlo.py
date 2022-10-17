from collections import Counter

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            probs = np.zeros(self.n_actions) + (epsilon / (self.n_actions - 1))
            a_star = argmax(self.Q_sa[s])
            probs[a_star] = 1 - epsilon
            return np.random.choice(self.n_actions, p=probs)
            # greedy = np.random.rand() > epsilon
            # return argmax(self.Q_sa) if greedy else np.random.choice(self.n_actions)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            return np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

        raise KeyError('Provide valid policy method')

    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        g_t_next = 0
        t_ep = len(actions)

        for i in range(t_ep):
            g_t = rewards[i] + self.gamma * g_t_next
            g_t_next = g_t
            self.Q_sa[states[i], actions[i]] = self.Q_sa[states[i], actions[i]] + self.learning_rate \
                                               * (g_t - self.Q_sa[states[i], actions[i]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    all_rewards, states, actions, rewards = [], [], [], []

    state = env.reset()
    states.append(state)
    done = False
    n_used = 0
    episode_length = 0

    while n_used < n_timesteps:
        if episode_length < max_episode_length and not done:
            episode_length += 1
            action = pi.select_action(state, epsilon=epsilon, policy=policy, temp=temp)
            s_next, reward, done = env.step(action)
            states.append(s_next)
            actions.append(action)
            rewards.append(reward)
            all_rewards.append(reward)
            state = s_next
            n_used += 1

        else:
            pi.update(states, actions, rewards)
            episode_length = 0
            done = False
            states, actions, rewards = [], [], []
            if plot and n_used % 1000 == 0:
                env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                           step_pause=0.1)  # Plot the Q-value estimates during n-step Q-learning execution
            state = env.reset()
            states.append(state)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                   step_pause=30)  # Plot the Q-value estimates during n-step Q-learning execution
    return all_rewards


def test():
    n_timesteps = 20_000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                          policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(Counter(rewards)))


if __name__ == '__main__':
    test()
