from trading_env import DataSource, TradingEnv, TradeBroker
from model import ReplayBuffer


def test_buffer_working(
	csv_file, 
	start_index=0, 
	end_index=10000, 
	window_size=25, 
	max_shares_held=10,
	buffer_size=1000
):
	
	# Initialize the env and other necessary objects
	d = DataSource(csv_file=csv_file, reframe=5, start_index=start_index, end_index=end_index)
	tb = TradeBroker(data_source=d, initial_balance=1000, price_history_length=window_size, max_shares_held=max_shares_held)
	env = TradingEnv(trading_broker=tb, observation_shape="2d")
	state = env.reset()
	
	# Initialize the replay buffer
	buffer = ReplayBuffer(buffer_size)
	done = False
	
	while not done:
		action = env.action_space.sample()
		next_state, reward, done, info = env.step(action)
		buffer.add(state, action, reward, next_state, done)
		state = next_state
		
	return buffer
		
		
	