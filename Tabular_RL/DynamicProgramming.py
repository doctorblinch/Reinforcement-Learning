import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    """ Class to store the Q-value iteration solution, perform updates, and select the greedy action """

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''
        # TO DO: Add own code
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = argmax(self.Q_sa[s])  # np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        self.Q_sa[s, a] = np.sum(
            p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1))
        )


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    iter = 0

    while True:
        delta = 0

        for s_index, state in enumerate(QIagent.Q_sa):
            for a_index, q_value in enumerate(state):
                QIagent.update(s_index, a_index, *env.model(s_index, a_index))  # TODO may fail here
                delta = max(delta, np.abs(q_value - QIagent.Q_sa[s_index, a_index]))

        # env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        # if iter % 2 == 1:
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        input("Q-value iteration, iteration {}, max error {}".format(iter, delta))
        # print("Q-value iteration, iteration {}, max error {}".format(iter, delta))

        if delta < threshold:
            break

        iter += 1

    # Plot current Q-value estimates & print max error
    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))


if __name__ == '__main__':
    experiment()
