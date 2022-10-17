import argparse
import json
import multiprocessing as mp
import os.path
import random
import time

import numpy as np

from configuration import Config
from training import Experiment

EXPERIMENT_REPETITION_VALUE = 5


def init_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='DQN',
    )
    parser.add_argument('--no_experience_replay', action='store_false', default=True)
    parser.add_argument('--no_target_network', action='store_false', default=True)
    parser.add_argument('--experiments_parameters')
    parser.add_argument('--experiments_quantity', default=5)
    parser.add_argument('-f', '--filename', required=False, type=str, default='hpo_config.json')
    return parser.parse_args()


def random_search(parameters: dict, n_samples: int):
    return [{k: random.sample(v, 1)[0] for k, v in parameters.items()} for _ in
            range(n_samples)] * EXPERIMENT_REPETITION_VALUE


def parse_config_file(filename: str = 'hpo_config.json'):
    with open(args.filename) as file:
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

    return (parameters.get('experiments_name'),
            parameters.get('experiments_count'),
            parameters.get('result_folder_name'),
            random_search(parsed_ranges, parameters.get('experiments_count'))
            )


def run_experiment(exp: Experiment):
    print(f'Starting experiment with {exp.config}')
    start_time = time.time()
    run_res = exp.run()
    print(f'Experiment {exp.config} finished in {round((time.time() - start_time) / 60, 2)} minutes')

    return run_res


def main(parser=None):
    if parser is None:
        config = Config()
        experiment = Experiment('ambulation', config, parser.no_target_network, parser.no_experience_replay)
        return experiment.run()

    experiment_name, experiments_count, result_folder_name, hyperparameter_grid = parse_config_file(parser.filename)

    n_threads = min(mp.cpu_count() // 4, len(hyperparameter_grid))
    print(f'Threads cnt = {n_threads}')
    pool = mp.Pool(n_threads)

    if not os.path.exists('models/'):
        os.mkdir('models')

    experiments = [Experiment(config=Config(conf), experiment_name=f'experiment_name_{index}',
                              use_target_net=parser.no_target_network,
                              use_experience_replay=parser.no_experience_replay) for index, conf in
                   enumerate(hyperparameter_grid)]
    results = pool.map(run_experiment, experiments)  # starmap
    print(results)


if __name__ == '__main__':
    args = init_argparse()
    main(args)
