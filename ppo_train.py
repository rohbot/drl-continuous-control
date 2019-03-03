# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
import os
import random
import numpy as np
from unityagents import UnityEnvironment
import pickle
import time
from model import ActorCritic
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


    
def test_env(env, model, device):
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



def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        

def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in (range(PPO_EPOCHS)):
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
            
            count_steps += 1
    
    writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)
    


if __name__ == "__main__":
    
    
    ts = str(int(time.time()))
    writer = SummaryWriter(comment="ppo_" + ts)
    #writer.reset()
    #os.system("mosquitto_pub -h 10.0.0.1 -t drl/log -m 'New Run: '" + ts)
    #os.system("mosquitto_pub -h 10.0.0.1 -t test -m 'New Run:' + ts")
    
    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # Prepare environments
    #envs = [make_env() for i in range(NUM_ENVS)]
    #envs = SubprocVecEnv(envs)
    env = UnityEnvironment(file_name=ENV_FILE)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations

    num_inputs  = env_info.vector_observations.shape[1]
    num_outputs = brain.vector_action_space_size

    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    #print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

    frame_idx  = 0
    train_epoch = 0
    best_reward = None

    #state = states[0]
    #print(state)
    early_stop = False
    all_scores = []
    averages = []

    
    for _ in tqdm(range(300)):

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []

        for _ in (range(PPO_STEPS)):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            #print(action)
            env_info = env.step(action.cpu().detach().numpy())[brain_name]
        
            #env_info = env.step(action)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = np.array([1 if d else 0 for d in env_info.local_done])
            
            log_prob = dist.log_prob(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            frame_idx += 1
                
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        advantage = normalize(advantage)

        ppo_update(frame_idx, states, actions, log_probs, returns, advantage)
        train_epoch += 1
        writer.add_scalar("episode", train_epoch, frame_idx)
        #writer.add_scalar("average", np.mean(returns), frame_idx)
    

        test_reward = test_env(env, model, device) 
        all_scores.append(test_reward)
        last_average = np.mean(np.array(all_scores[-100:])) if len(all_scores) > 100 else np.mean(np.array(all_scores))
        averages.append(last_average)
        writer.add_scalar("score", test_reward, frame_idx)
        writer.add_scalar("average", last_average, frame_idx)

        log_entry = 'Episode %s. reward: %.3f ave: %.3f' % (train_epoch, test_reward, last_average)
        print(log_entry)

        # Save a checkpoint every time we achieve a best reward
        if best_reward is None or best_reward < last_average:
            if best_reward is not None:
                writer.add_scalar("best_reward", test_reward, frame_idx)
                print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                name = "%s_best_%+.3f_%d.dat" % ("ppo", test_reward, frame_idx)
                fname = os.path.join('.', 'checkpoints', name)
                torch.save(model.state_dict(), fname)
            best_reward = test_reward
        if last_average > TARGET_REWARD:
            print("Solved Enviroment in %s epochs" % train_epoch)
            early_stop = True
            break

    
    
    # Save scores to file to graph later            
    timestamp =  str(int(time.time()))
    pickle.dump( all_scores, open( "scores/all_scores_"+timestamp+".p", "wb" ) )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig('ppo-30.png')