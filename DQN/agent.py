import json
import os
import random
from collections import deque

import numpy as np
import torch

from network import Qnet


# Import the environment.
# env = gym.make('CartPole-v1')

# Hyperparameters
# memory_size = Config.memory_size
# batch_size = Config.batch_size


# Build a memory for the agent.
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


# Hyperparameters
# gamma = Config.gamma
# e_max = Config.e_max
# e_min = Config.e_min
# e_decay = Config.e_decay


# Definition the agent itself
class Agent:
    def __init__(self, config, use_target_net=True, use_experience_replay=True):
        self.config = config
        self.memory = Memory(self.config.memory_size)
        self.epsilon = config.e_max

        self.use_target_net = use_target_net
        self.use_experience_replay = use_experience_replay

        self.policy_network = Qnet()
        if self.use_target_net:
            self.target_network = Qnet()

    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            # return env.action_space.sample()
            return random.randrange(2)

        state = torch.tensor(observation).float().detach()
        state = state.to(self.policy_network.device)
        state = state.unsqueeze(0)
        q_values = self.policy_network(state)
        return torch.argmax(q_values).item()

    def choose_action_softmax(self, state):
        state = torch.tensor(state).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        q_values = np.array(self.policy_network(state).data[0])
        return np.random.choice(len(q_values), p=softmax(q_values, self.config.temp))

    def learn(self):
        if len(self.memory.buffer) < self.config.batch_size:
            return

        if self.use_experience_replay:
            batch = self.memory.sample(self.config.batch_size)
        else:
            batch = self.memory.sample(len(self.memory.buffer))

        states = torch.tensor(np.array([each[0] for each in batch]), dtype=torch.float32).to(self.policy_network.device)
        actions = torch.tensor(np.array([each[1] for each in batch]), dtype=torch.long).to(self.policy_network.device)
        rewards = torch.tensor(np.array([each[2] for each in batch]), dtype=torch.float32).to(
            self.policy_network.device)
        states_next = torch.tensor(np.array([each[3] for each in batch]), dtype=torch.float32).to(
            self.policy_network.device)
        dones = torch.tensor(np.array([each[4] for each in batch]), dtype=torch.bool).to(self.policy_network.device)

        if self.use_experience_replay:
            batch_indices = np.arange(self.config.batch_size, dtype=np.int64)
        else:
            batch_indices = np.arange(len(self.memory.buffer), dtype=np.int64)
            self.memory.buffer.clear()

        q_values = self.policy_network(states)

        if self.use_target_net:
            next_q_values = self.target_network(states_next)
        else:
            next_q_values = self.policy_network(states_next)

        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        q_target = rewards + self.config.gamma * predicted_value_of_future * dones

        loss = self.policy_network.loss(q_target, predicted_value_of_now)
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

    def update_epsilon(self):
        self.epsilon *= self.config.e_decay
        self.epsilon = max(self.config.e_min, self.epsilon)

    def update_policy_net(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def load_training(self, run_name):
        self.policy_network.load_state_dict(torch.load(f'models/{run_name}/policy'))

    def save_models(self, run_name, rewards, config, validation_result):
        if not os.path.exists(os.path.join('models', run_name)):
            os.mkdir(os.path.join('models', run_name))

        torch.save(self.policy_network.state_dict(), os.path.join('models', run_name, 'policy'))
        np.array(rewards).tofile(os.path.join('models', run_name, 'rewards.csv'), sep='\n')
        config_json = {'training_config': config.__dict__,
                       'validation_result': validation_result}
        with open(os.path.join('models', run_name, 'config.json'), "w") as outfile:
            json.dump(config_json, outfile, indent=2)


def softmax(self, x, temp):
    x = x / temp
    z = x - max(x)
    return np.exp(z) / np.sum(np.exp(z))
