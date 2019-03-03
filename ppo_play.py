import os
import random
import numpy as np
from unityagents import UnityEnvironment
import time
import torch
from model import ActorCritic

ENV_FILE            = "./Reacher_Linux_NoVis/Reacher.x86_64"
ENV_FILE            = "./Reacher_Linux/Reacher.x86_64"
HIDDEN_SIZE         = 512


    
def test_env(env, model,device):
    env_info = env.reset(train_mode=True)[brain_name]    
    states = env_info.vector_observations                 
    scores = np.zeros(len(env_info.agents))                         
    while True:
        states = torch.FloatTensor(states).unsqueeze(0).to(device)
        dist, _ = model(states)
        action = dist.mean.detach().cpu().numpy()[0] # if deterministic \
        
        env_info = env.step(action)[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)

if __name__ == "__main__":    

    env = UnityEnvironment(file_name=ENV_FILE)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations

    num_inputs  = env_info.vector_observations.shape[1]
    num_outputs = brain.vector_action_space_size

    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load("models/ppo_trained.pth"))
    reward = test_env(env, model, device)
    print('reward: %s' % ( reward))

