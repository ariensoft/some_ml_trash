import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)


class BitmexSimple(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_size=30, train_mode=True):
        super(BitmexSimple, self).__init__()
        # self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self.action_space = spaces.Discrete(3)
        self.train_mode = train_mode
        self.data = self._load_data()
        self.data_max = self.data.max()
        self.data_min = self.data.min()
        self.max_reward = (self.data_max['askPrice'] - self.data_min['bidPrice'])
        self.current_reward = 0.0
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(window_size + 1, 6),
                                            dtype=np.float64)
        self.window_size = window_size
        self.current_step = 0
        self.current_data = pd.Series(dtype='float64')
        self.order_state = [0, 0.0]
        self.on_start_score = 0
        self.trades_made = [0, 0]
        self.history = []

    def _load_data(self):
        df = pd.read_csv("../data/bitmex_1m_2016-2020.csv")
        df = df.drop(['symbol', 'timestamp'], axis=1)
        df.dropna(inplace=True)
        df['fMA'] = df.bidPrice.rolling(window=7).mean()
        df['sMA'] = df.bidPrice.rolling(window=30).mean()
        # df['pct'] = df.bidPrice.pct_change()
        # df['pctSum'] = df.pct.cumsum()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        if self.train_mode:
            data = df.iloc[-int((len(df) / 4) * 3):]
        else:
            data = df.iloc[:int((len(df) / 4))]
        data.reset_index(inplace=True)
        data = data.drop(['level_0', 'index'], axis=1)
        return data

    def reset(self):
        self.current_step = 0
        self.order_state = [0, 0.0]
        self.on_start_score = 0
        self.trades_made = [0, 0]
        self.history = []
        self.current_reward = 0.0
        observation = self._next_observation()
        return observation

    def _next_observation(self):
        data = self.data.iloc[self.current_step:self.current_step + self.window_size]
        # bidSize, bidPrice, askPrice, askSize, fMA, sMA, = 6
        subset = data.values.copy()
        z = 0
        for s in subset:
            c = 0
            for v in s:
                subset[z][c] = v / self.data_max[c]
                c += 1
            z += 1
        observation = np.append(subset, [[
            self.order_state[0],
            self.current_reward,
            self.on_start_score,
            0,
            0,
            0,
        ]], axis=0)
        self.current_data = data.iloc[-1]
        return observation

    def _next_observation_single(self):
        data = self.data.iloc[self.current_step]
        # bidSize, bidPrice, askPrice, askSize, fMA, sMA, = 6
        vals = data.values.copy()
        obs = np.append(vals, [self.order_state[0]], axis=0)
        # print(obs.shape, obs)
        self.current_data = data
        return obs

    def step(self, action):
        prev_order_state = self.order_state[:]
        curr_action = self._take_action(action)
        reward = 0.0
        if self.order_state[0] == 0:
            reward -= 0.00069
        else:
            reward += (self.current_data['bidPrice'] - prev_order_state[1])
        if prev_order_state[0] > self.order_state[0]:
            if prev_order_state[1] <= self.order_state[1]:
                self.trades_made[0] += 1.0
            else:
                self.trades_made[1] += 1.0
            self.order_state[1] = 0.0
        self.on_start_score += reward
        self.current_reward = reward
        done = self.current_step >= len(self.data) - self.window_size - 1 or self.on_start_score <= 0
        self.current_step += 1
        observation = self._next_observation()
        info = {}
        return observation, reward, done, info  # obs reward done info

    def _take_action(self, action):
        history = [self.current_data['askPrice'], self.current_data['bidPrice'], 0, 0]
        if action <= 1:
            if self.order_state[0] == 0:
                self.order_state[0] = 1
                self.order_state[1] = self.current_data['askPrice']
                history[2] = self.current_data['askPrice']
        elif action <= 2:
            if self.order_state[0] == 1:
                self.order_state[0] = 0
                self.order_state[1] = self.current_data['bidPrice']
                history[3] = self.current_data['bidPrice']
        else:
            pass
        if not self.train_mode:
            self.history.append(history)
        return action

    def render(self, mode='human'):
        # print(f'Step: {self.current_step}')
        # print(f'Score: {self.on_start_score}')
        # print(f'trades_made: {self.trades_made}')
        # print(f'order_state: {self.order_state}')
        # print(f'Trades: {self.current_data}')
        data = np.array(self.history)
        buys = np.ma.masked_where(data[:, 2] == 0, data[:, 2])
        exits = np.ma.masked_where(data[:, 3] == 0, data[:, 3])
        plt.plot(data[:, 0])
        plt.plot(data[:, 1])
        plt.plot(buys, 'bo')
        plt.plot(exits, 'ro')
        plt.show()
        # return self.history
    
    def close(self):
        pass
