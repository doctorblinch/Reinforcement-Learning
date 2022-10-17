#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""
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
            # a = np.random.randint(0, self.n_actions)  # Replace this with correct action selection
            probs = np.zeros(self.n_actions) + (epsilon / (self.n_actions - 1))
            a_star = argmax(self.Q_sa[s])
            probs[a_star] = 1 - epsilon
            return np.random.choice(self.n_actions, p=probs)


        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # TO DO: Add own code
            # a = np.random.randint(0, self.n_actions)  # Replace this with correct action selection
            return np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

        raise KeyError('Provide valid policy method')
        # return a

    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        # pass
        t_ep = len(actions)

        for time in range(t_ep):
            mv = min(t_ep - time - 1, self.n)
            if done and time + mv + 1 == t_ep:
                gt = np.sum([rewards[time + 1] * self.gamma ** power for power in range(mv)])
            else:
                gt = np.sum([rewards[time + power] * (self.gamma ** power) for power in range(mv)]
                            ) + self.gamma ** mv * max(self.Q_sa[states[time + mv]])

            self.Q_sa[states[time], actions[time]] = (
                    self.Q_sa[states[time], actions[time]] +
                    self.learning_rate * (gt - self.Q_sa[states[time], actions[time]])
            )


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    state = env.reset()

    rewards = []
    rewards_history = []
    states = [state]
    actions = []
    done = False
    used_budget = 0
    episode_length = 0

    # TO DO: Write your n-step Q-learning algorithm here!
    # while n_timesteps > used_budget:

    while n_timesteps != 0:
        s = env.reset()  # reset state
        rew = []  # re-initialize episode rewards
        states = []  # re-initialize episode states
        actions = []  # re-initialize episode actions
        states.append(s)
        # collect episode
        for t in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rew.append(r)
            n_timesteps -= 1
            if done or n_timesteps == 0: break
        rewards += rew
        pi.update(states, actions, rew, done)
    #     if episode_length < max_episode_length and not done:
    #         episode_length += 1
    #         used_budget += 1
    #
    #         action = pi.select_action(state, epsilon=epsilon, policy=policy, temp=temp)
    #         actions.append(action)
    #
    #         state_next, reward, done = env.step(action)
    #         states.append(state_next)
    #         rewards.append(reward)
    #     else:
    #         NstepQLearningAgent.update(pi, states, actions, rewards, done)
    #         rewards_history += rewards
    #         rewards = []
    #         actions = []
    #         episode_length = 0
    #         done = False
    #
    #         if plot:
    #             env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
    #                        step_pause=0.1)  # Plot the Q-value estimates during n-step Q-learning execution
    #
    #         state = env.reset()
    #         states = [state]

    if plot and used_budget % 1000 == 0:
       env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=30) # Plot the Q-value estimates during n-step Q-learning execution

    # env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
    #            step_pause=30)  # Plot the Q-value estimates during n-step Q-learning execution
    return rewards


def test():
    n_timesteps = 20_000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = False

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                       policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(Counter(rewards)))


if __name__ == '__main__':
    test()
