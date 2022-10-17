from collections import Counter

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class QLearningAgent:
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
        # return a

    def update(self, s, a, r, s_next, done):
        # if done:
        #     return
        # TO DO: Add own code
        gt = r + self.gamma * np.max(self.Q_sa[s_next])
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (gt - self.Q_sa[s, a])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    """ runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestamp """

    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your Q-learning algorithm here!

    state = env.reset()
    for t in range(n_timesteps):

        action = pi.select_action(state, policy, epsilon=epsilon, temp=temp)
        state_next, reward, done = env.step(action)
        pi.update(state, action, reward, state_next, done)
        rewards.append(reward)

        state = env.reset() if done else state_next
        # return rewards

        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                       step_pause=0.1)  # Plot the Q-value estimates during Q-learning execution

        # if plot and t % 200 == 0:
        #     if t == 19800:
        #         env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=100)
        #     env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    return rewards


def test():
    n_timesteps = 20_000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = False

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(Counter(rewards)))


if __name__ == '__main__':
    test()
