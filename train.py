from model import ReplayBuffer, QNetworkCNN
import torch
import torch.nn.functional as F
import trading_env


class Trainer:
	
	def __init__(self, env, buffer_size=10000, batch_size=32, gamma=0.99, learning_rate=0.001, target_update=10, device="cpu"):
		self.env = env
		self.buffer = ReplayBuffer(buffer_size)
		self.batch_size = batch_size
		self.gamma = gamma
		self.target_update = target_update
		self.device = device
		
		# Initialize the Q-networks
		self.q_network = QNetworkCNN(env.observation_space.shape, env.action_space.n)
		self.q_network.to(device)
		self.target_network = QNetworkCNN(env.observation_space.shape, env.action_space.n)
		self.target_network.load_state_dict(self.q_network.state_dict())
		self.target_network.to(device)
		self.target_network.eval()
		self.rewards = []
		self.losses = []
		
		# Initialize the optimizer
		self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
		
		# Set the parameters for epsilon-greedy policy
		self.steps_done = 0
		self.eps_start = 0.9
		self.eps_end = 0.05
		self.eps_decay = 200
		
	def select_action(self, state):
		"""
		Basically we use something called an epsilon greedy policy.
		Where we take an random value between 0 and 1 and compare it with a threshold.
		This threshold starts at 0.9 and decays to 0.05 over time.
		If the random value is greater than the threshold, we take the action with the highest Q-value.
		Else we take a random action.
		At the beginning mahority of actions will be random but as the threshold decays, the number of random actions will decrease and we will start to take the action with the highest Q-value.
		:param state: 
		:return: 
		"""
		sample = torch.rand(1)
		eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
		                (1 - min(1.0, self.steps_done / self.eps_decay))
		self.steps_done += 1
		state = torch.tensor(state, dtype=torch.float32)
		if len(state.shape) == 2:
			state = state.unsqueeze(0)
		if len(state.shape) == 3:
			state = state.unsqueeze(0)
			
		state = state.to(self.device)
			
		if sample.item() > eps_threshold:
			with torch.no_grad():
				"""
				self.q_network.forward(state).max(1)[1].view(1, 1)
				means, lets say there are 3 actions, the output of the forward pass will be something like:
				[0.1, 0.2, 0.3]
				We take the max value, which is 0.3 and get the index of it, which is 2
				The .max function returns (max_value, index_of_max_value).
				The view function is used to reshape the tensor to the shape (1, 1)
				"""
				return self.q_network.forward(state).max(1)[1].view(1, 1)
		else:
			# Randomly select an action
			return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)
		
		
	def optimize_model(self, batch_size=32):
		
		# Select a random batch of values from the buffer
		if len(self.buffer) < batch_size:
			return
		states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
		
		# Convert the numpy arrays to torch tensors
		states = torch.tensor(states, dtype=torch.float32)
		states = states.unsqueeze(1)
		next_states = torch.tensor(next_states, dtype=torch.float32)
		next_states = next_states.unsqueeze(1)

		actions = torch.tensor(actions, dtype=torch.long)
		actions = actions.view(-1, 1)
		rewards = torch.tensor(rewards, dtype=torch.float32)
		rewards = rewards.to(self.device)
		
		states = states.to(self.device)
		q_s_a = self.q_network.forward(states).gather(1, actions) # For each state, get the Q-value of the action taken
		next_states = next_states.to(self.device)
		with torch.no_grad():
			q_s_a_max = self.target_network.forward(next_states).max(1)[0].detach()
		target = rewards + self.gamma * q_s_a_max * (1 - dones)
		target = target.unsqueeze(1)

		loss = F.smooth_l1_loss(q_s_a, target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.losses.append(loss.item())
		
	def train(self, num_episodes=1000):
		
		trading_env.logger.setLevel("CRITICAL")
		
		for episode in range(num_episodes):
			state = self.env.reset(random_start_index=True)
			done = False
			episode_reward = 0
			while not done:
				action = self.select_action(state)
				next_state, reward, done, info = self.env.step(action.item())
				episode_reward += reward
				state = state.detach().numpy()
				self.buffer.add(state, action, reward, next_state, done)
				state = next_state
				self.optimize_model()
				
				# Update the target network
				if self.steps_done % self.target_update == 0:
					self.target_network.load_state_dict(self.q_network.state_dict())
			print(f"Episode: {episode}, Reward: {episode_reward}")
