from collections import deque

import torch
import torch.nn as nn

import numpy as np

from configuration import Configuration
from networks import PolicyNet, CriticNet
# from experiment import Experiment


SUCCESS_SCORE = 475
MIN_RERUN = 5
MAX_RERUN = 50
LAST_N = 100


class Trajectory:
    def __init__(self, max_size=500):
        self.rewards = deque(maxlen=max_size)
        self.log_probabilities = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)


class StoppingCriteria:
    @staticmethod
    def check(experiment):
        if experiment.config.stopping_criteria is None:
            return False

        criteria2func = {
            'last_n': StoppingCriteria.last_n,
        }

        return criteria2func[experiment.config.stopping_criteria](experiment)

    @staticmethod
    def last_n(experiment, n=LAST_N, threshold=SUCCESS_SCORE):
        return sum(experiment.cumulative_rewards[-n:]) / n >= threshold

    @staticmethod
    def ready2rerun(experiment):
        return int(MAX_RERUN - (experiment.validation_score / SUCCESS_SCORE) * (MAX_RERUN - MIN_RERUN))


class AgentActorCritic:
    def __init__(self, config: Configuration):
        self.bootstrap = config.bootstrap
        self.baseline_subtract = config.baseline_subtract

        self.gamma = config.gamma
        self.depth = config.depth
        self.update_frequency = config.update_frequency

        self.policy_network = PolicyNet(config)
        self.critic_network = CriticNet(config)
        self.loss_function = nn.MSELoss()

        self.trajectory = Trajectory()

        self.actor_loss = deque(maxlen=self.update_frequency)
        self.critic_loss = deque(maxlen=self.update_frequency)

    def __str__(self):
        return f'<AgentActorCritic bootstrap={self.bootstrap}, baseline_subtract={self.baseline_subtract}>'

    def calculate_discounted_rewards(self):
        discounted_rewards = []
        rewards = list(self.trajectory.rewards)

        if self.bootstrap:
            for t in range(len(rewards)):
                T = min(self.depth, len(rewards) - t)
                discounts = [self.gamma ** i for i in range(T)]
                discounted_reward = np.sum([r * d for r, d in zip(rewards[t:t + T], discounts)])
                if not t + T == len(rewards):
                    discounted_reward += (self.gamma ** T) * self.trajectory.values[t + T]
                discounted_rewards.append(discounted_reward)
        else:
            for t in range(len(rewards)):
                discounts = [self.gamma ** i for i in range(len(rewards) - t)]
                discounted_reward = np.sum([r * d for r, d in zip(rewards[t:], discounts)])
                discounted_rewards.append(discounted_reward)
        return discounted_rewards

    def calculate_loss(self):
        discounted_rewards = torch.tensor(self.calculate_discounted_rewards(), dtype=torch.float32,
                                          device=self.policy_network.device)
        values = torch.cat(list(self.trajectory.values)).squeeze()
        log_probs = torch.cat(list(self.trajectory.log_probabilities)).squeeze()

        if self.baseline_subtract:
            advantages = discounted_rewards - values
            actor_loss = -torch.sum(log_probs * advantages.detach())

        else:
            actor_loss = -torch.sum(log_probs * discounted_rewards)

        critic_loss = self.loss_function(discounted_rewards, values)

        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)

        self.trajectory.rewards.clear()
        self.trajectory.log_probabilities.clear()
        self.trajectory.values.clear()

    def learn(self):
        actor_loss = torch.stack(list(self.actor_loss)).squeeze().mean()
        self.policy_network.optimizer.zero_grad()
        actor_loss.backward()
        self.policy_network.optimizer.step()
        self.policy_network.sigma *= 0.996  # Try annealing the exploration.

        critic_loss = torch.stack(list(self.critic_loss)).squeeze().mean()
        self.critic_network.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_network.optimizer.step()


class AgentReinforce:
    def __init__(self, config: Configuration):
        self.config = config
        self.update_frequency = config.update_frequency
        self.gamma = config.gamma

        self.policy_network = PolicyNet(config)
        self.trajectory = Trajectory()
        self.loss = deque(maxlen=self.update_frequency)

    def __str__(self):
        return f'<AgentReinforce>'

    def calculate_loss(self):
        rewards = torch.tensor(self.trajectory.rewards, dtype=torch.float32, device=self.policy_network.device)
        discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
        discounted_total_reward = np.sum([d * r for d, r in zip(discounts, rewards)])

        loss_storage = []
        for log_prob in self.trajectory.log_probabilities:
            loss_storage.append(-log_prob * discounted_total_reward)
        self.loss.append(torch.cat(loss_storage).sum())

        self.trajectory.rewards.clear()
        self.trajectory.log_probabilities.clear()

    def learn(self):
        loss = torch.mean(torch.stack([l for l in self.loss]))
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.policy_network.sigma *= 0.996  # Try annealing the exploration.