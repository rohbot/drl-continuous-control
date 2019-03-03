import os
import random
import numpy as np
from unityagents import UnityEnvironment
import time
import torch
from ppo_agent import Agent

ENV_FILE            = "./Reacher_Linux_NoVis/Reacher.x86_64"
#ENV_FILE            = "./Reacher_Linux/Reacher.x86_64"
WEIGHT_FILE         = "models/ppo_trained.pth"
HIDDEN_SIZE         = 512
  

if __name__ == "__main__":    

    env = UnityEnvironment(file_name=ENV_FILE)
    agent = Agent(env, HIDDEN_SIZE)
    agent.load_weights("models/ppo_trained.pth")
    reward = agent.play_episode()
    print('reward: %s' % ( reward))

