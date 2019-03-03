#!/usr/bin/env python3
import copy
import numpy as np

import torch
import torch.nn as nn

from unityagents import UnityEnvironment

#from tensorboardX import SummaryWriter
from mqtt_writer import SummaryWriter

import os
import pickle
import time


#mqtt_cmd = "mosquitto_pub -h 10.0.0.1 -t drl/"
env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

STATE_SIZE = env_info.vector_observations.shape[1]
ACTION_SIZE = brain.vector_action_space_size
BRAIN_NAME = env.brain_names[0]
NUM_AGENTS = len(env_info.agents)
NOISE_STD = 0.01
POPULATION_SIZE = 25
PARENTS_COUNT = int(POPULATION_SIZE / 5)



class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=128):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        #x = torch.Tensor(x)
        return self.mu(x)



def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        if done:
            break
    return reward


def evaluate(env, net):
    env_info = env.reset(train_mode=True)[BRAIN_NAME]    
    states = env_info.vector_observations                 
    scores = np.zeros(NUM_AGENTS)
    steps = 0
    while True:
        obs_v = torch.Tensor(states)
        #state = torch.from_numpy(states)
        actions = net(obs_v)
        env_info = env.step(actions.cpu().detach().numpy())[BRAIN_NAME]
        steps +=1
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    #writer.add_scalar("score", np.mean(scores), steps)
    return np.mean(scores)


def mutate_parent(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net


if __name__ == "__main__":
    writer = SummaryWriter(comment="-reacher-ga2")
    writer.reset()
    timestamp =  int(time.time())
    log_msg = "%d New Run  Pop size: %d Noise: %.3f  Parents: %d" %(timestamp, POPULATION_SIZE, NOISE_STD, PARENTS_COUNT )
    writer.log(log_msg)
    #env = gym.make("CartPole-v0")

    gen_idx = 0
    nets = [
        Net(STATE_SIZE, ACTION_SIZE)
        for _ in range(POPULATION_SIZE)
    ]
    population = [
        (net, evaluate(env, net))
        for net in nets
    ]
    prev_reward = 0
    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:PARENTS_COUNT]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)

        writer.add_scalar("episode", gen_idx, gen_idx)
        writer.add_scalar("average", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("score", reward_max, gen_idx)
        msg = "%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f" % (
            gen_idx, reward_mean, reward_max, reward_std)
        writer.log(msg)
        print(msg)
        if reward_mean > 30:
            print("Solved in %d steps" % gen_idx)
            break
        if reward_mean > prev_reward + 1:
            #save weights
            best_net = population[0][0]
            torch.save(best_net.state_dict(), f"models/ga-{POPULATION_SIZE}-{int(reward_mean)}.pth")
            prev_reward = reward_mean    

        # generate next population
        prev_population = population
        population = [population[0]]
        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent)
            fitness = evaluate(env, net)
            population.append((net, fitness))
        gen_idx += 1

    pass
