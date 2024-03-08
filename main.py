from trading_env import DataSource, TradingEnv, TradeBroker
from model import ReplayBuffer


def main():
	
	# Initialize the env and other necessary objects
	d = DataSource(csv_file="../Data/DAT_MT_EURUSD_M1_2021.csv", reframe=5, start_index=0, end_index=10000)
	tb = TradeBroker(data_source=d, initial_balance=1000, price_history_length=25)
	env = TradingEnv(trading_broker=tb, observation_shape="2d")
	state = env.reset()
	
	# Initialize the replay buffer
	buffer = ReplayBuffer(1000)
	
	while not env.is_done():
		action = env.action_space.sample()
		next_state, reward, done, info = env.step(action)
		buffer.add(state, action, reward, next_state, done)
		state = next_state
		
		
		
	