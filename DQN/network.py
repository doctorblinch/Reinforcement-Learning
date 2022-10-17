import torch
import torch.nn as nn

from configuration import Config


class Qnet(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()

        input_shape = config.observation_space
        output_shape = config.n_actions
        hidden_size = config.hidden_size
        learning_rate = config.learning_rate

        self.device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"
        self.input_shape = config.observation_space
        self.output_shape = output_shape

        self.input_layer = nn.Linear(self.input_shape, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.output_shape)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.input_layer(x))
        x = torch.nn.functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)

        return x
