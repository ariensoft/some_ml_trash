from abc import ABC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)


class BitstampEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, look_forward=500, train_mode=True):
        super(BitstampEnv, self).__init__()
        self.action_space = spaces.Box(np.array([0, 0]), np.array([4, 1]), shape=(2,))
        # side l/s/c/n, avail balance%
        self.train_mode = train_mode
        self.look_forward = look_forward
        self.data = self._load_data()
        self.max_steps = len(self.data)
        self.data_max_price = self.data.price.max()
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(int(self.look_forward + 6),))
        self.order = {'side': 0, 'contracts': 0, 'exec_price': 0.0, 'profit': 0.0, 'active': False}
        self.stats = []
        self.order_history = []
        self.balance = 0.0183
        self.current_step = 0

    def _load_data(self):
        bds = pd.read_csv("../data/bitstampUSDrepaired.csv")
        if self.train_mode:
            bds = bds.iloc[12000000:40000000]
        else:
            bds = bds.iloc[-40000000:]
        # bds['moneyFlow'] = bds.price * bds.volume
        bds['pct'] = bds.price.pct_change()
        bds['normPrice'] = bds.price / bds.price.max()
        bds.dropna(inplace=True)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        pcts = np.reshape(bds.pct.values, (-1, 1))
        bds['pctNorm'] = scaler.fit_transform(pcts)
        bds = bds.drop(['timestamp'], axis=1)
        return bds

    def reset(self):
        self.current_step = 0
        self.order['active'] = False
        self.order['profit'] = 0
        self.order_history = []
        self.stats = []
        self.balance = 0.0183
        self.current_step = 0
        observation = self._next_observation()
        return observation

    def _next_observation(self):
        self.stats = [-1, -1, -1, 1, self.balance, 0] # long, short, close, wait, balance, avail_contracts
        # available actions
        if not self.order['active']:
            self.stats[0] = 1
            self.stats[1] = 1
        else:
            self.stats[2] = 1
        self.curr_data = self.data.iloc[self.current_step:self.current_step + self.look_forward]
        obs = self.curr_data.normPrice
        obs = np.append(obs, self.stats)
        return obs

    def step(self, action):
        done = False
        reward = 0
        info = {}
        if self.current_step >= self.max_steps - self.look_forward - 1:
            info['terminal_reason'] = 'steps'
            info['terminal_balance'] = self.balance
            reward = self.balance
            done = True
        current_price = self.curr_data.iloc[0].price
        history = [current_price, 0, 0]
        contract_price = 1 / current_price
        affordable_contracts = int(self.balance / contract_price)
        self.stats[5] = affordable_contracts
        wtt_contracts = int((self.balance * action[1]) / contract_price)
        if int(np.ceil(action[0])) == 1:
            if not self.order['active']:
                if 0 < wtt_contracts <= affordable_contracts:
                    self.order = {'side': 0,
                                  'contracts': wtt_contracts,
                                  'exec_price': current_price,
                                  'profit': 0.0,
                                  'active': True}
                    self.balance -= wtt_contracts * contract_price
                    history[1] = current_price
        elif int(np.ceil(action[0])) == 2:
            if not self.order['active']:
                if 0 < wtt_contracts <= affordable_contracts:
                    self.order = {'side': 1,
                                  'contracts': wtt_contracts,
                                  'exec_price': current_price,
                                  'profit': 0.0,
                                  'active': True}
                    self.balance -= wtt_contracts * contract_price
                    history[1] = current_price
        elif int(np.ceil(action[0])) == 3:
            if self.order['active']:
                self.order['active'] = False
                history[2] = current_price
                self.balance += self.order['profit']
                self.order['profit'] = 0.0
        else:
            pass
        if self.order['active']:
            if self.order['side'] == 0:
                self.order['profit'] = ((1 / self.order['exec_price']) - (1 / current_price)) * self.order['contracts']
            else:
                self.order['profit'] = ((1 / current_price) - (1 / self.order['exec_price'])) * self.order['contracts']
        if self.balance + self.order['profit'] < contract_price and not self.order['active']:
            reward = -1000
            info['terminal_reason'] = 'bankrupt'
            info['terminal_balance'] = self.balance
            info['latest_profit'] = self.order['profit']
            done = True
        if self.balance + self.order['profit'] >= 1:
            info['terminal_reason'] = 'above 1btc'
            info['terminal_balance'] = self.balance
            info['latest_profit'] = self.order['profit']
            reward = self.balance + self.order['profit']
            done = True
        if not self.train_mode:
            self.order_history.append(history)
        self.current_step += 1
        observation = self._next_observation()
        info['reward'] = reward
        return observation, reward, done, info  # obs reward done info

    def render(self, mode='humans'):
        if self.order['active']:
            print(self.current_step, self.order, self.balance)
        else:
            print(self.current_step, self.balance)