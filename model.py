import numpy as np
from numpy.random import Generator, PCG64
from collections import deque

# Set the seed for the random number generator
np.random.seed(0)

class ReplayBuffer:
	
	def __init__(self, buffer_size=10000):
		self.memory = deque(maxlen=buffer_size)
		self.generator = Generator(PCG64(0))
		
	def reset(self):
		self.memory.clear()
	
	def add(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
	
	def sample(self, batch_size):
		indices = self.generator.choice(len(self.memory), batch_size, replace=False)
		states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
	
	def __len__(self):
		return len(self.memory)
