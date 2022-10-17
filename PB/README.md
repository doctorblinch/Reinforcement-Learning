# Policy-based Reinforcement Learning

By: **Kutay Nazli, Filip Jatelnicki, Ivan Horokhovskyi**

## Files description

- agent.py - contains trajectory, stopping criteria, both reinforce and actor-critic agents classes are located
- configuration.py - contains Config class which is used during an experiment to set hyperparameters for a training procedure and an agent 
- main.py - file which starts HPO or a single run with given/default parameters, configuration file parsing and parameters sampling and parallelization 
- networks.py - contains PyTorch policy and critic networks classes
- experiment.py - contains Experiment class with main training loop and validation process, can be executed to start an experiment with default parameters 
- req.txt - requirements file with libraries and their versions for a python environment
- hpo_config.json - configuration file which contains hyperparameters, their ranges and distribution to sample from for HPO
- actorcritic_best.json -  configuration file which contains selected (best) parameters which were picked after HPO for actor-critic network
- reinforce_best.json -  configuration file which contains selected (best) parameters which were picked after HPO for REINFORCE network
- advanced_analytics.ipynb - notebook which contains functions for getting statistics and plots from experiments
- ablation_graphs.ipynb - notebook which contains plots from experiments related tyo ablation study

Due to the submission file extension restriction **before running, please rename following files** _actorcritic_best.json.txt_ -> _actorcritic_best.json_,
_hpo_config.json.txt_ -> _hpo_config.json_ and _reinforce_best.json.txt_ -> _reinforce_best.json_.

## How to execute

### Start experiment

To execute a single experiment with default (from configuration.py) parameters: 

``python main.py --agent actor_critic``

Take into the account that _agent_ flag is optional (_actor_critic_ (default value) or _reinforce_)

---

To start an HPO with various parameters and using parallelization 
(take into the account that by default program will take all available cores of a computer)
it is required to provide a configuration file or use default one (hpo_config.json). The structure
of a configuration file is following: you should provide "experiments_name": string, 
"experiments_count": integer (note that this number will be multiplied by EXPERIMENT_REPETITION_VALUE 
constant which equals to 5 by default; this is done to do EXPERIMENT_REPETITION_VALUE repetitions to 
have more statistically significant results), "result_folder_name": string and "parameters" which is 
another key-value list (dictionary) which will contains hyperparameters. In this dict values can be 
constant for instance "episodes": 10000 or they can be followed by some rules like "learning_rate": {
      "min": 0.0001,
      "max": 0.05,
      "distribution": "log",
      "n": 5
    }
where "min" and "max" - are the lower and upper bound of sampling, "distribution" is type of sampling from 
the aforementioned bounds and "n" is the number of points that the space would be devided into.

``python main.py --filename hpo_config.json``

_filename_ flag corresponds to the configuration file for hyperparameters sampling. Validation is executed in experiments automatically
