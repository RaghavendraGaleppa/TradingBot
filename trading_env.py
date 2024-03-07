from collections import deque
from enum import Enum

from utils import get_data, get_logger
import gymnasium as gym
import numpy as np
import pandas as pd

logger = get_logger(__name__)

### Use below code to turn off the logging
# logger.setLevel("CRITICAL")


class Actions(Enum):
	Buy = 0
	Sell = 1
	Hold = 2

class DataSource:
	"""
	- This class holds the price data that it gets from a csv file. We will use this class to simulate an api which 
	provides us with the price data.
	- Methods of this class
		- reset: The index of the data will be reset to 0
		- get_next_price: This method will return the next price data and the index of the data
		- get_idx_price: This method will return the price data at the given index
		- is_done: This method will return True if the index is at the end of the data
		- __len__: This method will return the length of the data
	"""
	
	def __init__(self, csv_file, reframe=5, start_index=0, end_index=None):
		self.data = get_data(csv_file, reframe)
		end_index = len(self.data) if end_index is None else end_index
		self.current_index = 0
		self.data = self.data[start_index:end_index]
		self.max_index = len(self.data)
		
	def reset(self):
		self.current_index = 0
		
	def get_next_price(self):
		if self.current_index == self.max_index:
			logger.debug("No more data to return")
			return None
		price = self.data.iloc[self.current_index]
		self.current_index += 1
		return price, self.current_index
	
	def get_idx_price(self, idx):
		return self.data.iloc[idx]
	
	def is_done(self):
		return self.current_index == self.max_index
	
	def __len__(self):
		return self.max_index
	
	
	
class TradeBroker:
	
	def __init__(self, data_source: DataSource, initial_balance=1000, price_history_length=100):
		self.data_source = data_source
		self.data_source.reset()
		
		self.initial_balance = initial_balance
		self.balance = initial_balance
		self.price_history = deque(maxlen=price_history_length)
		self.trade_history = []
		self.shares_held = 0
		
		# Some meta information
		self.balance_history = []
		self.total_buys = 0
		self.total_sells = 0
		self.total_holds = 0
		
		
	def reset(self):
		self.data_source.reset()
		self.balance = self.initial_balance
		self.price_history.clear() # This will empty the deque
		self.trade_history.clear() # This will empty the list
		self.shares_held = 0
		self.balance_history.clear()
		self.total_buys = 0
		self.total_sells = 0
		
	def is_done(self):
		if self.balance <= 0 and self.shares_held == 0:
			return True
		return self.data_source.is_done()
		
	def buy(self):
		"""
		- This method will buy 1 share of the stock at the current price
		:return: None
		"""
		price, idx = self.data_source.get_next_price()
		self.price_history.append(price)
		if price is None:
			logger.debug("No more data to buy")
			return False
		
		if self.balance <= 0 and self.shares_held == 0:
			logger.debug("No balance to buy")
			return False
		
		if self.balance < price["close"]:
			logger.debug(f"Not enough balance to buy at price: {price['close']}")
			return False
		
		self.balance -= price["close"]
		self.shares_held += 1
		self.trade_history.append({"action": "buy", "price": price["close"], "idx": idx})
		self.total_buys += 1
		self.balance_history.append(self.balance)
		logger.debug(f"Buying 1 share at {price['close']}")
		return True
	
	def sell(self):
		"""
		- This method will sell 1 share of the stock at the current price
		:return: None
		"""
		price, idx = self.data_source.get_next_price()
		self.price_history.append(price)
		if price is None:
			logger.debug("No more data to sell")
			return False
		
		if self.shares_held == 0:
			logger.debug("No shares to sell")
			return False
		
		self.balance += price["close"]
		self.shares_held -= 1
		self.trade_history.append({"action": "sell", "price": price["close"], "idx": idx})
		self.total_sells += 1
		self.balance_history.append(self.balance)
		logger.debug(f"Selling 1 share at {price['close']}")
		return True
	
	def hold(self):
		"""
		- This method will do nothing
		:return: None
		"""
		price, idx = self.data_source.get_next_price()
		self.price_history.append(price)
		if price is None:
			logger.debug("No more data to hold")
			return False
		logger.debug(f"Holding at {price['close']}")
		self.total_holds += 1
		self.balance_history.append(self.balance)
		return True
	
	def get_price_history(self):
		"""
		- This method will return the price history
		:return: list of price history
		"""
		price_data = pd.DataFrame(self.price_history)
		if len(price_data) == 0:
			return price_data
		price_data['close_prev'] = price_data['close'].shift(1)
		price_data.dropna(inplace=True)
		price_data['rate_of_change'] = (price_data['close'] - price_data['close_prev']) / price_data['close_prev']
		return price_data
	
	def _get_meta_info(self):
		"""
		- This method will return the meta information
		:return: dict
		"""
		return {
			"total_buys": self.total_buys,
			"total_sells": self.total_sells,
			"total_holds": self.total_holds
		}
	
	def get_current_assets(self):
		return {
			"balance": self.balance,
			"shares_held": self.shares_held,
			"current_price": self.price_history[-1]["close"] if self.price_history else None,
			"P/L": self.balance - self.initial_balance
		}




class TradingEnv(gym.Env):
	
	def __init__(self, trading_broker, observation_shape="1d"):
		self.trading_broker = trading_broker
		self.action_space = gym.spaces.Discrete(len(Actions))
		if observation_shape == "1d":
			self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.trading_broker.price_history.maxlen, 1), dtype=float)
		elif observation_shape == "2d":
			self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.trading_broker.price_history.maxlen, 4), dtype=float)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.trading_broker.price_history.maxlen, 2), dtype=float)
		self.done = False
		self.last_P_L = 0
		
		
	def reset(self):
		self.trading_broker.reset()
		self.done = False
		self.last_P_L = 0
		return self._get_observation()
	
	def step(self, action):
		if self.done is True:
			return self._get_observation(), 0, self.done, {}
		if action == Actions.Buy.value:
			res = self.trading_broker.buy()
		elif action == Actions.Sell.value:
			res = self.trading_broker.sell()
		else:
			res = self.trading_broker.hold()
		
		self.done = self.trading_broker.is_done()
		obs = self._get_observation()
		reward = self._calculate_reward()
		info = {
			"action": action,
			"action_res": res,
			"assets": self.trading_broker.get_current_assets()
		}
		return obs, reward, self.done, info
	
	def _get_observation(self):
		price_data = self.trading_broker.get_price_history()
		if len(price_data) == 0:
			return np.zeros(self.observation_space.shape)
		obs = price_data[['open', 'high', 'low', 'close']].to_numpy()
		obs = obs / obs.max()
		return obs
		
	
	def _calculate_reward(self):
		"""
		Reward should be calculated in terms of how much profit or loss we have made since the last step.
		It should be positive if current P/L is higher than the last P/L
		else it should be negative
		:return: reward: float
		"""
		p_l = self.trading_broker.get_current_assets()["P/L"]
		reward = p_l - self.last_P_L
		self.last_P_L = p_l
		return reward