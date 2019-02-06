from comet_ml import Experiment

from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent
import os
import pickle
import time

mqtt_cmd = "mosquitto_pub -h 10.0.0.1 -t drl/"

#env = UnityEnvironment(file_name='20_agents/Reacher_Linux_NoVis/Reacher.x86_64')
env = UnityEnvironment(file_name='one_agent/Reacher_Linux_NoVis/Reacher.x86_64')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
states = env_info.vector_observations                  # get the current state (for each agent)
agent = Agent(state_size=states.shape[1], action_size=brain.vector_action_space_size, random_seed=10)

experiment = Experiment(project_name="reacher")

def ddpg(n_episodes=2000, max_t=70):
    scores_deque = deque(maxlen=100)
    scores_global = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        #print(states)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        scores_average = 0
        #score = 0
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states                               # roll over states to next time step
            scores += rewards                                  # update the score (for each agent)            
            if np.any(dones):                                  # exit loop if episode finished
                break       
        score = np.mean(scores)
        scores_deque.append(score)
        scores_global.append(score)
        score_average = np.mean(scores_deque)
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
		
        os.system(mqtt_cmd +  "episode -m " + str(i_episode))
        os.system(mqtt_cmd + "score -m {:.2f}".format(score))
        os.system(mqtt_cmd + "average -m {:.2f}".format(np.mean(score_average)))
        experiment.log_current_epoch(i_episode)
        experiment.log_metric("score", score, step=i_episode)
        experiment.log_metric("score_average", score_average, step=i_episode)

    return scores_global

scores = ddpg()