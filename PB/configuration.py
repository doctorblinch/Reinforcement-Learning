import inspect


class Configuration:
    # RL
    gamma = 1
    learning_rate = 0.001
    sigma = 0.1
    update_frequency = 10
    depth = 100
    budget = 10000
    baseline_subtract = True
    bootstrap = True
    stopping_criteria = 'last_n'

    # NN
    hidden_size = 64
    device = 'cpu'

    # env
    observation_space = 4
    n_actions = 2
    save_folder = 'model'
    validation_iterations = 1_000
    render = False

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
        return f'<Config {self.get_non_default_parameters()}>'

        # return f'<Config [lr={self.learning_rate}, bs={self.batch_size}, ed={self.e_decay}, tf={self.target_fraction}, log={self.log}]>'

    def to_dict(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        return {a[0]: a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))}

    def get_non_default_parameters(self):
        d = self.to_dict()
        default_conf = Configuration().to_dict()
        non_default = {}

        for key in d.keys():
            if key not in default_conf or d[key] != default_conf[key]:
                non_default[key] = d[key]

        return non_default

