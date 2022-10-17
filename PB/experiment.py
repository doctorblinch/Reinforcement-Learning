import csv
import json
import os.path
from time import time

import gym
import torch

from agent import AgentActorCritic, AgentReinforce, StoppingCriteria
from configuration import Configuration

PRINT_FREQ = 50
VALIDATION_SUCCESS_SCORE = 475
EXPERIMENT_FOLDER_NAME = 'exp'
VALIDATION_LIMIT = 4


class Experiment:
    def __init__(self, config: Configuration = None, agent: str = 'actor_critic'):
        agent2class = {
            'reinforce': AgentReinforce,
            'actor_critic': AgentActorCritic,
        }
        assert agent in agent2class, f'Please pick a valid agent {list(agent2class.values())}'

        self.config = config if config is not None else Configuration()
        self.agent = agent2class[agent](self.config)
        self.env = gym.make('CartPole-v1')

        # Stats
        self.cumulative_rewards = []
        self.val_cumulative_rewards = []
        self.experiment_time = -1
        self.validation_score = -1

    def run(self):
        start_time = time()

        if type(self.agent) == AgentReinforce:
            self._run_reinforce()
            self.experiment_time = time() - start_time
            return

        elif type(self.agent) == AgentActorCritic:
            self._run_agent_critic()
            self.experiment_time = time() - start_time
            return

        assert False, f'Invalid run type!'

    def validate(self):
        print(f'Starting validation for {self.config}')
        self.val_cumulative_rewards = []

        for i in range(self.config.validation_iterations):
            state = self.env.reset()
            cum_reward = 0

            while True:
                if self.config.render:
                    self.env.render()

                action = self.agent.policy_network.choose_action(state)[0]
                state_new, reward, done, _ = self.env.step(action)
                state = state_new
                cum_reward += reward

                if done:
                    if i % 100 == 0 and i != 0:
                        print(f'Validation {self.config} iteration {i}')
                    self.val_cumulative_rewards.append(cum_reward)
                    break

        self.validation_score = sum(self.val_cumulative_rewards) / len(self.val_cumulative_rewards)
        print(
            f'Validation {self.config} {"successful" if self.validation_score >= VALIDATION_SUCCESS_SCORE else f"failed with {self.validation_score} validation score"}')
        return self.validation_score >= VALIDATION_SUCCESS_SCORE

    def _run_reinforce(self):
        last_validation = 0
        validation_count = 0

        for episode in range(self.config.budget):
            state = self.env.reset()
            cumulative_reward = 0

            while True:
                # env.render()
                action, log_probability = self.agent.policy_network.choose_action(state)
                state_next, reward, done, _ = self.env.step(action)
                self.agent.trajectory.rewards.append(reward)
                self.agent.trajectory.log_probabilities.append(log_probability)
                state = state_next
                cumulative_reward += reward

                if done:
                    self.agent.calculate_loss()
                    self.cumulative_rewards.append(cumulative_reward)
                    if episode % self.agent.update_frequency == 0:
                        self.agent.learn()
                        if episode % PRINT_FREQ == 0:
                            print(
                                f'Episode {episode} Reward {round(sum(self.cumulative_rewards[-PRINT_FREQ:]) / PRINT_FREQ)} '
                            )
                    if StoppingCriteria.check(self) and episode - last_validation > StoppingCriteria.ready2rerun(self):
                        last_validation = episode
                        success = self.validate()
                        validation_count += 1

                        if success:
                            print(f'Validation successful with {self.validation_score} score in {self.config}')
                            self.save_results()
                            return
                        if validation_count >= VALIDATION_LIMIT:
                            print(
                                f'Validation failed after {validation_count} attempts in {self.config}')
                            return

                    break

    def _run_agent_critic(self):
        last_validation = 0
        validation_count = 0

        for episode in range(self.config.budget):
            state = self.env.reset()
            cumulative_reward = 0

            while True:
                # env.render()
                action, log_probability = self.agent.policy_network.choose_action(state)
                value = self.agent.critic_network.critic_value(state)
                state_next, reward, done, _ = self.env.step(action)
                self.agent.trajectory.rewards.append(reward)
                self.agent.trajectory.log_probabilities.append(log_probability)
                self.agent.trajectory.values.append(value)
                state = state_next
                cumulative_reward += reward

                if done:
                    self.agent.calculate_loss()
                    self.cumulative_rewards.append(cumulative_reward)
                    if episode % self.agent.update_frequency == 0 and not episode == 0:
                        self.agent.learn()
                        if episode % PRINT_FREQ == 0:
                            print(
                                f'Episode {episode} Reward {round(sum(self.cumulative_rewards[-PRINT_FREQ:]) / PRINT_FREQ)} '
                            )  # .format(int(episode/agent.update_frequency), cumulative_reward))

                    if StoppingCriteria.check(self) and episode - last_validation > StoppingCriteria.ready2rerun(self):
                        last_validation = episode
                        success = self.validate()
                        validation_count += 1

                        if success:
                            print(f'Validation successful with {self.validation_score} score in {self.config}')
                            self.save_results()
                            return
                        if validation_count >= VALIDATION_LIMIT:
                            print(
                                f'Validation failed after {validation_count} attempts in {self.config}')
                            return
                    break

    def save_results(self):
        if len(self.cumulative_rewards) == 0:
            assert False, 'Trying to save results before running the experiment!'

        save_folder = self.config.save_folder
        if not os.path.exists(save_folder):
            os.mkdir(self.config.save_folder)

        folders = [d for d in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, d))]
        new_index = max(
            [int(folder[len(EXPERIMENT_FOLDER_NAME) + 1:].split('_')[0])
             for folder in folders if folder.startswith(EXPERIMENT_FOLDER_NAME)]
        ) + 1 if len(folders) > 0 else 1

        new_folder = os.path.join(save_folder, f'{EXPERIMENT_FOLDER_NAME}_{new_index}')
        os.mkdir(new_folder)

        # Save cumulative_rewards
        with open(os.path.join(new_folder, 'rewards.csv'), 'w') as f:
            write = csv.writer(f)
            write.writerows([[i] for i in self.cumulative_rewards])

        with open(os.path.join(new_folder, 'validation_rewards.csv'), 'w') as f:
            write = csv.writer(f)
            write.writerows([[i] for i in self.val_cumulative_rewards])

        # Save models
        torch.save(self.agent.policy_network.state_dict(), os.path.join(new_folder, 'policy'))
        if type(self.agent) == AgentActorCritic:
            torch.save(self.agent.critic_network.state_dict(), os.path.join(new_folder, 'critic'))

        # Save config
        config_dict = self.config.to_dict()
        config_dict['validation_score'] = self.validation_score
        with open(os.path.join(new_folder, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


if __name__ == '__main__':
    # reinforce | actor_critic
    experiment = Experiment(agent='reinforce')
    experiment.run()
