from abc import ABC
from pyts.multivariate.image import JointRecurrencePlot
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces
from OrderGod import Orders

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)


class Bitmex(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size=30, episode_len=30, train_mode=True):
        super(Bitmex, self).__init__()
        self.action_space = spaces.Box(np.array([0, 0]), np.array([3, 1]), shape=(2,), dtype=np.float64)
        self.train_mode = train_mode
        self.window_size = window_size
        self.episode_len = int(episode_len * 1440)
        self.data = self._load_data()
        self.subset = np.array([])
        self.transformed_data = []
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(int(self.window_size * self.window_size),),
                                            dtype=np.float64)
        self.orders = Orders()
        self.order_history = []
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
        df['pct'] = df.bidPrice.pct_change()
        df['pctSum'] = df.pct.cumsum()
        df.dropna(inplace=True)
        df = df.drop(['bidPrice', 'pct', 'symbol', 'timestamp', 'bidSize', 'askPrice', 'askSize'], axis=1)
        if self.train_mode:
            data = df.iloc[-int((len(df) / 4) * 3):]
        else:
            data = df.iloc[:int((len(df) / 4))]
        return data.values  # bidSize, bidPrice, askPrice, askSize, fMA, sMA, pct, pctSum = 8

    def _next_observation(self):
        obs = self.transformed_data[self.current_step].flatten()
        return obs

    def reset(self):
        self.current_step = 0
        self.orders = Orders()
        self.order_history = []
        start = int(np.random.uniform(0, len(self.data) - self.episode_len))
        self.subset = self.data[start:start + self.episode_len]
        self.transformed_data = self._window_transform(self.window_size)
        observation = self._next_observation()
        return observation

    def step(self, action):
        reward = 0
        done = False
        if self.current_step >= len(self.subset) - self.window_size:
            done = True
        current_price = self.subset[self.current_step + self.window_size - 1]
        self.current_step += 1
        observation = self._next_observation()
        info = {}
        return observation, reward, done, info  # obs reward done info

    def render(self, mode='human'):
        pass


