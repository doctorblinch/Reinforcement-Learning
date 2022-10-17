# Reinforcement Learning, Assignment: Deep Q Learning

By: **Kutay Nazli, Ivan Horokhovskyi, Filip Jatelnicki**

## Files description

- agent.py - contains memory (replay buffer) and agent classes along with affiliated methods like learning procedure and action selection
- configuration.py - contains Config class which is used during an experiment to set hyperparameters for a training procedure and an agent 
- main.py - file which starts HPO or a single run with given/default parameters, configuration file parsing and parameters sampling and parallelization 
- network.py - contains PyTorch network class
- training.py - contains Experiment class with main training loop and validation process, can be executed to start an experiment with default parameters 
- validate.py - script to validate an agent/saved weights to get average values during environment and check if environment is solved by an agent
- req.txt - requirements file with libraries and their versions for a python environment
- hpo_config.json - configuration file which contains hyperparameters, their ranges and distribution to sample from for HPO  
- ambulation.json -  configuration file which contains selected (best) parameters which were picked after HPO
- Baby'sSecondDQN_OnSteroids.ipynb - initial notebook which was later generalized and extended to framework
- Analysis.ipynb - notebook which contains functions for getting statistics and plots from experiments
- policy - weights of the model which perfectly solves the environment (reward 500) in each run

Due to the submission file extension restriction **before running, please rename following files** _policy.txt_ -> _policy_,
_hpo_config.json.txt_ -> _hpo_config.json_ and _ambulation.json.txt_ -> _ambulation.json_.

## How to execute 

### Create environment and install dependencies

``$ python -m venv venv``

``. venv/bin/activate``

``pip install -r req.txt``

### Start experiment 
To execute a single experiment with default (from configuration.py) parameters: 

``python training.py --no_experience_replay --no_target_net``

Take into the account that _no_experience_replay_ and _no_target_net_ flags are optional

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

_filename_ flag corresponds to the configuration file for hyperparameters sampling. In addition _no_experience_replay_ 
and _no_target_net_ flags are available.  

## Validation

Validation is executed in experiments automatically, however it can be done manually 
if user wants to validate/visualize a trained agent. Please put weights of a model
in following path: _models/{run_name}/policy_ where {run_name} is changed to any 
other name. For instance, you can take the provided model weights and put weights 
(_policy_ file) in a path: _models/validation_example/policy_. After that execute:

``python validate.py --run_name validation_example --render``

_render_ is an optional parameter and is used to render the gym environment which may
require additional libs installed on your computer and also takes longer to execute.