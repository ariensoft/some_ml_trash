from abc import ABC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)


class BitstampProphet(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size=60, prophecy_len=30, train_mode=True):
        super(BitstampProphet, self).__init__()
        self.action_space = spaces.Box(np.array([0, 0]), np.array([4, 1]), shape=(2,))
        # side l/s/c/n, avail balance%
        self.train_mode = train_mode
        self.window_size = window_size
        self.prophecy_len = prophecy_len
        self.data = self._load_data()
        self.curr_data = []
        self.data_max_price = self.data.price.max()
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(int(self.window_size + self.prophecy_len + 7),))
        self.order = {'side': 0, 'contracts': 0, 'exec_price': 0.0, 'profit': 0.0, 'active': False}
        self.order_history = []
        self.balance = 0.0183
        self.current_step = 0

    def _load_data(self):
        bds = pd.read_csv("../data/bitstampUSDrepaired.csv")
        if self.train_mode:
            bds = bds.iloc[12000000:40000000]
        else:
            bds = bds.iloc[40000000:]
        bds['moneyFlow'] = bds.price * bds.volume
        bds['pct'] = bds.price.pct_change() * 100
        bds['normPrice'] = bds.price / bds.price.max()
        bds.dropna(inplace=True)
        bds = bds.drop(['timestamp'], axis=1)
        return bds

    def reset(self):
        self.current_step = 0
        self.order['active'] = False
        self.order_history = []
        self.balance = 0.0183
        self.current_step = 0
        observation = self._next_observation()
        return observation

    def _next_observation(self):
        stats = [self.balance, 0, 0, 0, 0, 1, self.order['profit']]
        # available actions
        if self.order['active']:
            stats[4] = 1
            stats[5] = 1
        else:
            stats[3] = 1
        self.curr_data = self.data.iloc[self.current_step:self.current_step + self.window_size + self.prophecy_len]
        obs = self.curr_data.normPrice
        obs = np.append(obs, stats)
        return obs

    def step(self, action):
        done = False
        reward = 0
        info = {}
        if self.current_step == len(self.data) - (self.window_size + self.prophecy_len) - 1:
            info['terminal_reason'] = 'steps'
            info['terminal_balance'] = self.balance
            reward = self.balance
            done = True
        # print(self.curr_data.shape)
        current_price = self.curr_data.iloc[self.window_size - 1].price
        history = [current_price, 0, 0]
        contract_price = 1 / current_price
        contracts = int((self.balance * action[1]) / contract_price)
        if int(np.ceil(action[0])) == 1:
            if self.order['active']:
                reward += -0.0001
            else:
                if contracts > 0:
                    self.order = {'side': 0,
                                  'contracts': contracts,
                                  'exec_price': current_price,
                                  'profit': 0.0,
                                  'active': True}
                    self.balance -= contracts * contract_price
                    history[1] = current_price
                else:
                    reward += -0.0001
        elif int(np.ceil(action[0])) == 2:
            if self.order['active']:
                reward += -0.001
            else:
                if contracts > 0:
                    self.order = {'side': 1,
                                  'contracts': contracts,
                                  'exec_price': current_price,
                                  'profit': 0.0,
                                  'active': True}
                    self.balance -= contracts * contract_price
                    history[1] = current_price
                else:
                    reward += -0.0001
        elif int(np.ceil(action[0])) == 3:
            if self.order['active']:
                self.order['active'] = False
                history[2] = current_price
                self.balance += self.order['profit']
                reward += self.order['profit'] * 1000
                self.order['profit'] = 0.0
            else:
                reward += -0.0001
        else:
            pass
        if self.order['active']:
            if self.order['side'] == 0:
                self.order['profit'] = ((1 / self.order['exec_price']) - (1 / current_price)) * self.order['contracts']
            else:
                self.order['profit'] = ((1 / current_price) - (1 / self.order['exec_price'])) * self.order['contracts']
            self.balance += self.order['profit']
        if self.balance < contract_price:
            reward = -1000
            info['terminal_reason'] = 'bankrupt'
            info['terminal_balance'] = self.balance
            done = True
        if self.balance >= 1:
            info['terminal_reason'] = 'above 1btc'
            info['terminal_balance'] = self.balance
            reward = 1000
            done = True
        # print(reward)
        if not self.train_mode:
            self.order_history.append(history)
        self.current_step += 1
        observation = self._next_observation()
        info['reward'] = reward
        return observation, reward, done, info  # obs reward done info

    def render(self, mode='humans'):
        c = 0
        if self.order['active'] and c == 0:
            c = 1
            print(self.current_step, self.order, self.balance)
        if not self.order['active'] and c == 1:
            c = 0
            print(self.current_step, self.order, self.balance)
