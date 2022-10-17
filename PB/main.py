import argparse
import json
import multiprocessing as mp
import os.path
import random
import time

import numpy as np

from configuration import Configuration
from experiment import Experiment

# EXPERIMENT_REPETITION_VALUE = 5


def init_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Policy Based Reinforcement Learning',
    )
    parser.add_argument('--experiments_quantity', default=5)
    parser.add_argument('-f', '--filename', required=False, type=str)  # , default='hpo_config.json')
    parser.add_argument('-a', '--agent', required=False, type=str, default='actor_critic')
    return parser.parse_args()


def random_search(parameters: dict, n_samples: int, experiment_repetition_value: int):
    return [{k: random.sample(v, 1)[0] for k, v in parameters.items()} for _ in
            range(n_samples)] * experiment_repetition_value


def parse_config_file(filename: str = 'hpo_config.json'):
    with open(filename) as file:
        parameters = json.load(file)

    parsed_ranges = {}

    for key, val in parameters.get('parameters').items():
        v = None
        if type(val) == list:
            parsed_ranges[key] = val
        elif type(val) in [str, float, int, bool]:
            parsed_ranges[key] = [val]
        elif type(val) == dict:
            dist = val['distribution']
            assert dist in ('log', 'uniformal'), 'Config parsing error!\nUnknown distribution!'
            if dist == 'log':
                parsed_ranges[key] = np.geomspace(val['min'], val['max'], val['n']).tolist()
            elif dist == 'uniformal':
                parsed_ranges[key] = np.linspace(val['min'], val['max'], val['n']).tolist()

    return random_search(
        parsed_ranges, parameters.get('experiments_count'), parameters.get('experiment_repetition_value')
    ), parameters


def run_experiment(exp: Experiment):
    print(f'Starting experiment with {exp.config} and {exp.agent}')
    exp.run()
    print(f'Experiment {exp.config} finished in {round(exp.experiment_time / 60, 2)} minutes')

    return exp.cumulative_rewards, exp.config, exp.experiment_time


def main(parser=None):
    if parser is None or parser.filename is None:
        config = Configuration()
        agent = 'actor_critic'  # reinforce
        experiment = Experiment(config=config, agent=agent)
        print(f'Starting experiment with {experiment.config} and {experiment.agent}')
        experiment.run()
        print(f'Experiment {experiment.config} finished in {round(experiment.experiment_time / 60, 2)} minutes')
        return experiment

    hyperparameter_grid, parameters = parse_config_file(parser.filename)
    print(hyperparameter_grid)
    n_threads = min(mp.cpu_count() // 4, len(hyperparameter_grid))
    print(f'Threads cnt = {n_threads}')
    pool = mp.Pool(n_threads)

    experiments = [Experiment(config=Configuration(conf), agent=parser.agent)
                   for index, conf in enumerate(hyperparameter_grid)]
    results = pool.map(run_experiment, experiments)  # starmap
    # print(results)


if __name__ == '__main__':
    args = init_argparse()
    main(args)
