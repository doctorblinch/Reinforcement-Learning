import torch
import torch.nn as nn

from configuration import Configuration


class PolicyNet(torch.nn.Module):
    def __init__(self, config: Configuration):
        # learning_rate=0.001, sigma=0.1, hidden_size=64, device='cpu', observation_space=4, n_actions=2):
        super().__init__()
        self.config = config
        self.sigma = config.sigma
        self.learning_rate = config.learning_rate

        hidden_size = config.hidden_size
        output_shape = config.n_actions

        self.device = config.device
        self.input_shape = config.observation_space
        self.output_shape = output_shape

        self.input_layer = nn.Linear(self.input_shape, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.output_shape)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return torch.nn.functional.softmax(x, dim=1)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        probabilities = self.forward(state) + torch.normal(
            0.0, self.config.sigma, size=(1, 2)
        )
        probabilities[probabilities < 0] = 0

        model = torch.distributions.Categorical(probabilities)
        action = model.sample()
        return action.item(), model.log_prob(action)


class CriticNet(torch.nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        # TODO different config params for critic and policy nets
        self.learning_rate = config.learning_rate
        output_shape = 1

        self.device = config.device
        self.input_shape = config.observation_space
        hidden_size = config.hidden_size
        self.output_shape = output_shape

        self.input_layer = nn.Linear(self.input_shape, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.output_shape)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def critic_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.forward(state)
