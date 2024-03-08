import numpy as np
from numpy.random import Generator, PCG64
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the seed for the random number generator
np.random.seed(0)
torch.manual_seed(0)


class QNetworkCNN(nn.Module):
	
	def __init__(self, input_shape, n_actions):
		super(QNetworkCNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(64*input_shape[0]*input_shape[1], 512)
		self.fc2 = nn.Linear(512, n_actions)
	
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		return self.fc2(x)
	

class ReplayBuffer:
	
	def __init__(self, buffer_size=10000):
		self.memory = deque(maxlen=buffer_size)
		self.generator = Generator(PCG64(0))
		
	def reset(self):
		self.memory.clear()
	
	def add(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
		
	def __getitem__(self, item):
		return self.memory[item]
	
	def sample(self, batch_size):
		indices = self.generator.choice(len(self.memory), batch_size, replace=False)
		states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
	
	def __len__(self):
		return len(self.memory)
