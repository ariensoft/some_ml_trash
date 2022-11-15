from abc import ABC
from pyts.multivariate.image import JointRecurrencePlot
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)


class BitmexLimited(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size=30, episode_len=30, train_mode=True):
        super(BitmexLimited, self).__init__()
        self.action_space = spaces.Box(np.array([0, 0]), np.array([4, 1]), shape=(2,))
        # side l/s/c/n, avail balance%
        self.train_mode = train_mode
        self.window_size = window_size
        self.episode_len = int(episode_len * 1440)
        self.data = self._load_data()
        self.subset = np.array([])
        self.transformed_data = []
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(int(self.window_size * self.window_size + 7),))
        self.order = {'side': 0, 'contracts': 0, 'exec_price': 0.0, 'profit': 0.0, 'active': False}
        self.order_history = []
        self.balance = 0.0183
        self.avail_balance = self.balance
        self.current_step = 0

    def _cnn_transform(self, window=30):  # df = np.arr
        new_rows = []
        for row in range(self.subset.shape[0]):
            if row + window <= self.subset.shape[0]:
                subset = self.subset[row:row + window]
                tmp = []
                for v in range(self.subset.shape[1]):
                    tmp.append(subset[:, v])
                new_rows.append(tmp)
        res = np.asarray(new_rows)
        jrp = JointRecurrencePlot(threshold='point', percentage=50)
        x_jrp = jrp.fit_transform(res)
        return x_jrp

    def _window_transform(self, window=30):  # df = np.arr
        new_rows = []
        for row in range(self.subset.shape[0]):
            if row + window <= self.subset.shape[0]:
                new_rows.append(self.subset[row:row + window])
        n = np.asarray(new_rows)
        res = np.squeeze(n)
        rp = RecurrencePlot(threshold='point', percentage=30)
        X_rp = rp.fit_transform(res)
        return X_rp

    def _load_data(self):
        df = pd.read_csv("../data/bitmex_1m_2016-2020.csv")
        df.dropna(inplace=True)
        # df['pct'] = df.bidPrice.pct_change()
        # df['pctSum'] = df.pct.cumsum()
        df.dropna(inplace=True)
        df = df.drop(['symbol', 'timestamp', 'bidSize', 'askPrice', 'askSize'], axis=1)
        if self.train_mode:
            data = df.iloc[-int((len(df) / 4) * 3):]
        else:
            data = df.iloc[:int((len(df) / 4))]
        return data.values  # bidSize, bidPrice, askPrice, askSize, fMA, sMA, pct, pctSum = 8

    def _next_observation(self):
        stats = [self.balance, self.avail_balance, 0, 0, 0, 1, self.order['profit']]
        # available actions
        if self.order['active']:
            stats[4] = 1
            stats[5] = 1
        else:
            stats[3] = 1
        obs = self.transformed_data[self.current_step]
        obs = np.append(obs, stats)
        return obs

    def reset(self):
        self.current_step = 0
        self.order['active'] = False
        self.order_history = []
        self.balance = 0.0183
        self.avail_balance = self.balance
        start = int(np.random.uniform(0, len(self.data) - self.episode_len))
        self.subset = self.data[start:start + self.episode_len]
        self.transformed_data = self._window_transform(self.window_size)
        observation = self._next_observation()
        return observation

    def step(self, action):
        reward = 0
        done = False
        info = {}
        if self.current_step == len(self.subset) - self.window_size - 1:
            info['terminal_reason'] = 'steps'
            info['terminal_balance'] = self.balance
            reward += self.balance
            done = True
        current_price = self.subset[self.current_step + self.window_size - 1][0]
        history = [current_price, 0, 0]
        contract_price = 1 / current_price
        contracts = int((self.avail_balance * action[1]) / contract_price)
        if int(np.ceil(action[0])) == 1:
            if self.order['active']:
                reward += -0.001
            else:
                if contracts > 0:
                    self.order = {'side': 0,
                                  'contracts': contracts,
                                  'exec_price': current_price,
                                  'profit': 0.0,
                                  'active': True}
                    self.avail_balance -= contracts * contract_price
                    history[1] = current_price
                    reward += 0.001
                else:
                    reward += -0.001
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
                    self.avail_balance -= contracts * contract_price
                    history[1] = current_price
                    reward += 0.001
                else:
                    reward += -0.001
        elif int(np.ceil(action[0])) == 3:
            if self.order['active']:
                self.order['active'] = False
                history[2] = current_price
                self.balance += self.order['profit']
                self.avail_balance = self.balance
                reward += self.order['profit'] * 1000
                self.order['profit'] = 0.0
            else:
                reward += -0.001
        else:
            pass
        if self.order['active']:
            if self.order['side'] == 0:
                self.order['profit'] = ((1 / self.order['exec_price']) - (1 / current_price)) * self.order['contracts']
            else:
                self.order['profit'] = ((1 / current_price) - (1 / self.order['exec_price'])) * self.order['contracts']
            self.balance += self.order['profit']
            reward += self.order['profit']
        if self.balance < contract_price and not self.order['active']:
            reward = -1
            info['terminal_reason'] = 'bankrupt'
            info['terminal_balance'] = self.balance
            done = True
        if self.balance >= 1:
            info['terminal_reason'] = 'above 1btc'
            info['terminal_balance'] = self.balance
            reward = 0.01
            done = True
        # print(reward)
        if not self.train_mode:
            self.order_history.append(history)
        self.current_step += 1
        observation = self._next_observation()
        info['reward'] = reward
        return observation, reward, done, info  # obs reward done info

    def render(self, mode='humans'):
        data = np.array(self.order_history)
        buys = np.ma.masked_where(data[:, 1] == 0, data[:, 1])
        exits = np.ma.masked_where(data[:, 2] == 0, data[:, 2])
        plt.plot(data[:, 0])
        plt.plot(buys, 'bo')
        plt.plot(exits, 'ro')
        plt.show()
        print('Balance', self.balance)
