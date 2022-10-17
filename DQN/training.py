import argparse
import time
from collections import deque

import gym
import numpy as np
import wandb

from agent import Agent
from configuration import Config
from validate import validate


class Experiment:
    def __init__(self, experiment_name, config: Config = None, use_target_net=True, use_experience_replay=True):
        if config is None:
            config = Config()

        config.batch_size = int(config.batch_size)
        self.config = config
        self.env = gym.make('CartPole-v1')
        self.episodes = config.episodes
        self.target_fraction = config.target_fraction
        self.agent = Agent(config, use_target_net, use_experience_replay)
        self.rewards = []
        self.last_rewards = deque(maxlen=config.num_considered_rewards)
        self.log = config.log
        self.wandb_log = config.wandb_log
        self.experiment_name = experiment_name

    def run(self) -> bool:
        if self.wandb_log is True:
            wandb.init(project="dqn", entity="piwo", group="test", name=str(time.time()), config={
                "episodes": self.config.episodes,
                "hidden_size": self.config.hidden_size,
                "learning_rate": self.config.learning_rate,
                "memory_size": self.config.memory_size,
                "batch_size": self.config.batch_size,
                "gamma": self.config.gamma,
                "e_max": self.config.e_max,
                "e_min": self.config.e_min,
                "e_decay": self.config.e_decay,
                "target_fraction": self.config.target_fraction,
                "num_considered_rewards": self.config.num_considered_rewards,
                "learning_reward_threshold": self.config.learning_reward_threshold
            })
        for episode in range(self.episodes):
            state = self.env.reset()

            cumulative_reward = 0
            while True:
                # env.render()
                action = self.agent.choose_action(state)
                state_next, reward, done, _ = self.env.step(action)
                self.agent.memory.add((state, action, reward, state_next, 1 - done))
                state = state_next
                cumulative_reward += reward

                if done:
                    if episode % self.target_fraction == 0 and self.agent.use_target_net:
                        self.agent.update_policy_net()
                    self.agent.learn()
                    self.agent.update_epsilon()
                    if episode % 500 == 0:
                        print("{} Episode {} Reward {} Epsilon {} Conf {}".format(self.experiment_name, episode,
                                                                                  np.mean(
                                                                                      self.last_rewards) if self.last_rewards else 0,
                                                                                  self.agent.epsilon, self.config))
                    self.last_rewards.append(cumulative_reward)
                    self.rewards.append(cumulative_reward)
                    if self.wandb_log:
                        wandb.log(
                            {'reward': cumulative_reward, 'last_rewards': np.mean(self.last_rewards)}, step=episode)
                    break

            if np.mean(self.last_rewards) > self.config.learning_reward_threshold:
                print(f"Model {self.experiment_name} trained \n Last rewards: {self.last_rewards}")
                if self.log:
                    validation_result = validate(agent_config=self.agent.config, run_name=self.experiment_name,
                                                 render=False, log=False,
                                                 weights=self.agent.policy_network.state_dict())
                    self.agent.save_models(run_name=self.experiment_name, rewards=self.rewards, config=self.config,
                                           validation_result=validation_result)
                    # After done training see, whether can solve environment once

                    if self.wandb_log:
                        wandb.finish()
                return True
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_target_net', action='store_false', default=True)
    parser.add_argument('--no_experience_replay', action='store_false', default=True)
    args = parser.parse_args()

    experiment = Experiment('test_training', use_target_net=args.no_target_net,
                            use_experience_replay=args.no_experience_replay)
    experiment.run()
