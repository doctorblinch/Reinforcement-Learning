from collections import Counter

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class NstepQLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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

    def update(self, states, actions, rewards, done):
        """ states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state """
        # TO DO: Add own code
        t_ep = len(actions)
        for t in range(t_ep):
            s, a = states[t], actions[t]
            m = min(self.n, t_ep - t)
            if done[t + m - 1]:
                g = np.sum(np.array([(self.gamma ** i) * rewards[t + i] for i in range(m)]))
            else:
                g = np.sum(np.array([(self.gamma ** i) * rewards[t + i] for i in range(m)])) + (self.gamma ** m) * max(
                    self.Q_sa[states[t + m]])

            self.Q_sa[s][a] += self.learning_rate * (g - self.Q_sa[s][a])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    """ runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep """

    # Make environment
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []
    count = 0

    while n_timesteps > 0:
        s_next = env.reset()
        dones, states, actions, episode_rewards = [], [], [], []
        states.append(s_next)
        for t in range(min(max_episode_length, n_timesteps)):
            action = pi.select_action(s=s_next, policy=policy, epsilon=epsilon, temp=temp)
            actions.append(action)
            s_next, reward, done = env.step(action)
            if plot and n_timesteps % 1000 == 0:
                env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.5)  # Plot the Q-value
            states.append(s_next)
            dones.append(done)
            episode_rewards.append(reward)
            rewards.append(reward)
            n_timesteps -= 1
            if done:
                count += 1
                break
        pi.update(states=states, actions=actions, rewards=episode_rewards, done=dones)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=100)

    return rewards


def test():
    n_timesteps = 20_000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'softmax'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                       policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(Counter(rewards)))


if __name__ == '__main__':
    test()
