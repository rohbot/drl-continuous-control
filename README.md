[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control

### Introduction

For this project, a Deep Reinforcement Learning agent will be trained to work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Getting Started

1. Clone repository 

`git clone https://github.com/rohbot/drl-continuous-control.git`

2. Install Dependencies

`pip install -r requirements.txt`

3. Download the environment from one of the links below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Training Agent
To start training, run [*ppo_train.py*](ppo_train.py) in terminal window:

`python ppo_train.py`


### Running Trained Agent
To see a trained agent in action simply open [*Continuous_Control.ipynb*](Continuous_Control.ipynb) in Jupyter Notebook and follow the instructions there:

Start Jupyter Notebook

`jupyter notebook`

Trained model weights are included to see the agent solved the Unity ML Reacher Environment. 



