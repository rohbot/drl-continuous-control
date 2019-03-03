import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from model import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
	"""Interacts with and learns from the environment."""
	
	def __init__(self, env, hidden_size=512, lr=3e-4, gamma=0.99, tau=.95, 
		ppo_steps=20, ppo_epochs=10, mini_batch_size=5, 
		ppo_clip=0.2, gradient_clip=5):
		"""Initialize an Agent object.

		Params
		======
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			random_seed (int): random seed
			
		"""
		self.env = env
		self.brain_name = env.brain_names[0]
		brain = env.brains[self.brain_name]
		env_info = env.reset(train_mode=True)[self.brain_name]   		
		self.state_size  = env_info.vector_observations.shape[1]
		self.action_size = brain.vector_action_space_size

		self.model = ActorCritic(self.state_size, self.action_size, hidden_size).to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


		self.gamma = gamma
		self.tau = tau
		self.ppo_epochs = ppo_epochs
		self.mini_batch_size = mini_batch_size
		self.ppo_steps =ppo_steps
		self.mini_batch_size  = mini_batch_size
		self.ppo_epochs = ppo_epochs
		self.ppo_clip = ppo_clip
		self.gradient_clip = gradient_clip
		
		self.frame_idx = 0
		self.episode = 0


	def load_weights(self, weights_file):
		print("Loading weights: ",weights_file)
		self.model.load_state_dict(torch.load(weights_file))
		

	def play_episode(self):
		env_info = self.env.reset(train_mode=True)[self.brain_name]    
		states = env_info.vector_observations                 
		scores = np.zeros(len(env_info.agents))                         
		while True:
			states = torch.FloatTensor(states).unsqueeze(0).to(device)
			dist, _ = self.model(states)
			action = dist.mean.detach().cpu().numpy()[0] 
			
			env_info = self.env.step(action)[self.brain_name]
			next_states = env_info.vector_observations         
			rewards = env_info.rewards                         
			dones = env_info.local_done                     
			scores += env_info.rewards                      
			states = next_states                               
			if np.any(dones):                                  
				break
		
		return np.mean(scores)	
	
		

	def compute_gae(self, next_value, rewards, masks, values):

		values = values + [next_value]
		gae = 0
		returns = []
		for step in reversed(range(len(rewards))):
			delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
			gae = delta + self.gamma * self.tau * masks[step] * gae
			returns.insert(0, gae + values[step])
		return returns

	def ppo_iter(self, states, actions, log_probs, returns, advantage):
		batch_size = states.size(0)
		for _ in range(batch_size // self.mini_batch_size):
			rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
			yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
			

	def ppo_update(self,  states, actions, log_probs, returns, advantages):
		
		for _ in range(self.ppo_epochs):
			for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
				dist, value = self.model(state)
				entropy = dist.entropy().mean()
				new_log_probs = dist.log_prob(action)

				ratio = (new_log_probs - old_log_probs).exp()
				surr1 = ratio * advantage
				surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantage

				actor_loss  = - torch.min(surr1, surr2).mean()
				critic_loss = (return_ - value).pow(2).mean()

				loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
				self.optimizer.step()


	def learn(self):

		log_probs = []
		values    = []
		states    = []
		actions   = []
		rewards   = []
		masks     = []
		
		env_info = self.env.reset(train_mode=True)[self.brain_name]
		state = env_info.vector_observations

		for _ in (range(self.ppo_steps)):
			state = torch.FloatTensor(state).to(device)
			dist, value = self.model(state)

			action = dist.sample()
			#print(action)
			env_info = self.env.step(action.cpu().detach().numpy())[self.brain_name]
		
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
			self.frame_idx += 1
				
		next_state = torch.FloatTensor(next_state).to(device)
		_, next_value = self.model(next_state)
		returns = self.compute_gae(next_value, rewards, masks, values)

		returns   = torch.cat(returns).detach()
		log_probs = torch.cat(log_probs).detach()
		values    = torch.cat(values).detach()
		states    = torch.cat(states)
		actions   = torch.cat(actions)
		advantage = returns - values
		# Normalize Advantage
		advantage -= advantage.mean()
		advantage /= (advantage.std() + 1e-8)

		self.ppo_update(states, actions, log_probs, returns, advantage)
		self.episode += 1
 

