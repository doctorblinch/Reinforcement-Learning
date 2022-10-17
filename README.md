# Reinforcement-Learning

## Value-based

[Value-based report link](https://github.com/doctorblinch/Reinforcement-Learning/blob/main/Tabular_RL/RL-Tabular.pdf)

This part was dedicated to some standard methods for value-based tabular reinforcement learning namely Q-learning, SARSA, n-step Q-learning and Monte-Carlo methods. In addition, these algorithms were compared to classic dynamic-programming approach. After that hyperparameter tuning, exploration/exploitation tradeoff was discussed, on/off-policy algorithms comparison experiments and their analysis were done.

## Deep Q-Learning

[DQL report link](https://github.com/doctorblinch/Reinforcement-Learning/blob/main/DQN/DQN_RL.pdf)

As the systems in Reinforcement Learning become high-dimensional, continuous and have exponential branching factor, the methodology moves away from the tabular approach into implementing deep learning methods to reach solutions. We consider the simple CartPole-v1 environment in Gym and implement a Deep Q-Network (DQN) to perform Q-value iteration to solve the environment, along with an automated parallelizable framework for solving reinforcement learning problems with HPO and subsequent analysis. We present the methodology for the network architecture, the hyperparameters, and optimization strategies, and perform and discuss the results of an ablation study comparing the performance impact of various features of the DQN such as exploration strategies, experience replay and a target network.

## Policy-based

[DQL report link](https://github.com/doctorblinch/Reinforcement-Learning/blob/main/PB/Policy-based.pdf)

In Reinforcement Learning problems that have high-dimensional or continuous action spaces, the value based approach fails to replicate its performance in problems with smaller dimensions. Therefore, instead of learning the Q-value function and having a separate algorithm to pick the optimal next action, it is possible to learn instead the policy. This work considers the simple CartPole-v1 environment in Gym and implements the REINFORCE and actor-critic algorithms to perform gradient-based policy search to solve the environment. Additionally, an ablation study is performed to compare the REINFORCE algorithm to the more sophisticated actor-critic approach, and to investigate the performance impact of bootstrapping and baseline-subtraction in the latter.
