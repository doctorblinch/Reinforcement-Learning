import argparse
from collections import deque

import gym
import numpy as np

import configuration
from agent import Agent


def validate(run_name, render, agent_config=None, log=True, weights=None):
    print("Start validation process")
    episodes = 1000
    target_fraction = 10
    episode_number = []
    episode_rewards = []
    episode_rewards_100 = deque(maxlen=100)

    if agent_config is None:
        agent_config = configuration.Config()

    agent = Agent(agent_config)
    env = gym.make('CartPole-v1')

    if weights is None:
        agent.load_training(run_name)
    else:
        agent.policy_network.load_state_dict(weights)

    agent.epsilon = 0
    for i in range(episodes):
        state = env.reset()
        # print(state)
        # state = np.reshape(state, [1, input_shape])

        # Keep track of cumulative reward.
        cum_reward = 0

        while True:
            if render:
                env.render()

            action = agent.choose_action(state)  # Choose next action according to the annealed e-greedy policy.
            state_, reward, done, _ = env.step(action)  # Act on it.
            # state_ = np.reshape(state_, [1, input_shape])
            state = state_  # Progress the state to the next.
            cum_reward += reward  # Increase the reward.

            if done:
                episode_number.append(i)
                episode_rewards.append(cum_reward)
                episode_rewards_100.append(cum_reward)

                if log:
                    print(
                        f"Episode {i} Mean reward_100 {np.mean(episode_rewards_100).round(2)} Epsilon {agent.epsilon}")
                break

        if len(episode_rewards_100) >= 100 and np.mean(episode_rewards_100) > configuration.Config.env_solved:
            print("Environment solved")
            return {'solver': True, 'min_value': min(episode_rewards), 'max_value': max(episode_rewards),
                    'mean_value': np.mean(episode_rewards)}

    return {'solver': False, 'min_value': min(episode_rewards), 'max_value': max(episode_rewards),
            'mean_value': np.mean(episode_rewards)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run_name')
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()
    validate(args.run_name, args.render)
