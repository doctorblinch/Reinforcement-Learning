import os


class Config:
    episodes = os.getenv('episodes', 10000)
    hidden_size = os.getenv('hidden_size', 64)
    learning_rate = os.getenv('learning_rate', 0.0001)  # Optimize. (???? Kutay is confused.)

    memory_size = os.getenv('memory_size', 10000)
    batch_size = os.getenv('batch_size', 1500)  # Optimize. (1000-2000)

    gamma = os.getenv('gamma', 0.95)
    e_max = os.getenv('e_max', 1.0)
    e_min = os.getenv('e_min', 0.001)
    e_decay = os.getenv('e_decay', 0.9995)  # Optimize. (0.995-0.9999)
    temp = os.getenv('temp', 0.8)

    target_fraction = os.getenv('target_fraction', 100)  # Optimize. (10-100)

    num_considered_rewards = os.getenv('num_considered_rewards', 5)
    learning_reward_threshold = os.getenv('learning_reward_threshold', 475)

    log = os.getenv('log', True)
    wandb_log = os.getenv('wandb_log', False)

    env_solved = os.getenv('env_solved', 475)
    observation_space = 4
    n_actions = 2
    non_default_params = {}

    def __init__(self, d: dict = None):
        if d is None:
            return

        for key, value in d.items():
            self.__setattr__(key, value)
            self.non_default_params[key] = value

    def __repr__(self):
        return f'<Config {[f"{key}: {value}" for key, value in self.non_default_params.items()]}>'

    def __str__(self):
        if self.non_default_params:
            return f'<Config {[f"{key}: {value}" for key, value in self.non_default_params.items()]}>'

        return f'<Config [lr={self.learning_rate}, bs={self.batch_size}, ed={self.e_decay}, tf={self.target_fraction}, log={self.log}]>'
