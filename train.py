
import os
import random
import numpy as np
from unityagents import UnityEnvironment
import pickle
import time
from model import ActorCritic
from ppo_agent import Agent
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from tqdm import tqdm

ENV_FILE            = "./Reacher_Linux_NoVis/Reacher.x86_64"
HIDDEN_SIZE         = 512
LEARNING_RATE       = 3e-4
GAMMA               = 0.99
GAE_TAU             = 0.95
PPO_CLIP            = 0.2
PPO_STEPS           = 2048
MINI_BATCH_SIZE     = 64
PPO_EPOCHS          = 10

TARGET_REWARD       = 30


if __name__ == "__main__":    

    env = UnityEnvironment(file_name=ENV_FILE)
    agent = Agent(env, HIDDEN_SIZE, LEARNING_RATE, GAMMA, GAE_TAU, PPO_STEPS, PPO_EPOCHS, MINI_BATCH_SIZE, PPO_CLIP)
    best_reward = None
    all_scores = []
    averages = []
    solved = False
    while not solved:
        agent.learn()
        test_reward = agent.play_episode() 
        all_scores.append(test_reward)
        last_average = np.mean(np.array(all_scores[-100:])) if len(all_scores) > 100 else np.mean(np.array(all_scores))
        averages.append(last_average)

        log_entry = 'Episode %s. reward: %.3f ave: %.3f' % (agent.episode, test_reward, last_average)
        print(log_entry)
        
        if best_reward is None or best_reward < test_reward:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                name = "%s_best_%+.3f_%d.dat" % ("ppo", test_reward, agent.episode)
                fname = os.path.join('.', 'checkpoints', name)
                torch.save(agent.model.state_dict(), fname)
            best_reward = test_reward
        if last_average > TARGET_REWARD:
            print("Solved Enviroment in %s episodes" % agent.episode)
            solved = True
            break
            
    #Save all_scores to file for graphing purposes        
    timestamp =  str(int(time.time()))
    pickle.dump( all_scores, open( "all_scores_"+timestamp+".p", "wb" ) )