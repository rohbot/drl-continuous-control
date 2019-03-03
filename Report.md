# Report

This project implements a reinforcement learning agent to controls a simulation of a robotic arm. The double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. 
The objective is  to an average score of +30 (over 100 consecutive episodes, and over all agents). 

The agents is trained using Proximal Policy Optimization (PPO) Algorithm based on the paper released by [OpenAI](https://blog.openai.com/openai-baselines-ppo/).

## Implementation

The implementation split into a few smaller modules: 

* model.py - Neural Network model implemented with PyTorch
* ppo_agent.py - PPO agent implementation as described in [paper](https://arxiv.org/abs/1707.06347) mentioned above
* ppo_train.py - imports all required modules and allows the enviroment to be explored and the agent trained
* ppo_play.py - Runs an Agent using pre-trained weights from ppo_train.py

## Learning Algorithm

The agent in this project utilised code for implementing the PPO alogirthm as outlined in the [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb) github repo.	


The agent comprises of a pair of neural networks, the actor and the critic networks.

 Each network has the same architecture of 3 fully connected layers with ReLu activation on the first two layers. It also implements a replay buffer to store the experiences of an action in the enviroment, that is later sampled in batches to train the neural nets.


1. It starts by initializing the replay buffer and inital weights for the neural networks.
1. For each episode within the max_episodes given it:
	1. Resets Environment
	1. Gets current state from enviroment
	For each step in maximum number of timesteps per episode:
		1. Picks an action using state using an epslion-greedy algorithm
		1. Executes this action in the enviroments to obtain rewards, next_state, done
		1. Stores this experience in the replay buffer
		If timestep matches EVERY_UPDATE 
			1. Sample random batch of experiences from replay buffer
			1. Get predicted Q values from target network using next_states
			1. Compute target for current states using rewards + (gamma * Q_targets_next * (1 - dones))
			1. Get expected values from local model using states and actions
			1. Compute MSE Loss with expected values and target values
			1. Minimize loss using Adam Optimization, backprop and step
			1. Perform soft update of local network with target network and TAU value
	1. Keep looping until the average score over last 100 episodes >= 13
		1. Save weights of local network to file 


### Hyperparameter Tuning

The initial parameters were set to the same values as in [Deep_Q_Network_Solution.ipynb](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb)

	HIDDEN_SIZE         = 512
	LEARNING_RATE       = 3e-4
	ADAM_EPS            = 1e-5
	GAMMA               = 0.99
	GAE_LAMBDA          = 0.95
	PPO_EPSILON         = 0.2
	CRITIC_DISCOUNT     = 0.5
	ENTROPY_BETA        = 0.001
	PPO_STEPS           = 2048
	MINI_BATCH_SIZE     = 64
	PPO_EPOCHS          = 10
	TARGET_REWARD       = 30

This leads to an agent that can achieve an average score of 30 over 100 episodes after 153 episodes

![Plot of Rewards](ppo-training.png)



## Results

--TODO


## Ideas for Future Work
---

* Implement an evolutionary system or genetic algorithm based on this [implementation](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter16). I initally modified this code to run with the Reacher environment [ga_train.py](archive/ga_train.py). However even after several days of training it failed to converge so went on to implement this PPO algorithm. Given more time I would like to delve deeper into this GA/ES approach with more tweaking of learning rates and possibly a better implemetation of cross-over and parent selection.

* Implement a Model-Based RL agent that uses Particle Swarm Optimizatiion [Reinforcement Learning with Particle Swarm Optimization Policy PSO-P in Continuous State and Action Spaces](https://pdfs.semanticscholar.org/d0c4/9a9ed109cb8573217a9a0affbad7881b77a4.pdf)
