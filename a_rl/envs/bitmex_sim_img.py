import io
from abc import ABC

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
import gym
from gym import spaces


class BitmexSimImg(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timeframe=30, window_size=30, train_mode=True):
        super(BitmexSimImg, self).__init__()
        # self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self.action_space = spaces.Discrete(3)
        self.train_mode = train_mode
        self.data = self._load_data()
        self.test = self._plot_window_to_np(self.data.iloc[:30])
        self.img_shape = self.test.shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2]),
                                            dtype=np.uint8)
        self.timeframe = timeframe * 24 * 60
        self.window_size = window_size
        self.start = 0
        self.current_step = 0
        self.current_data = pd.Series(dtype='float64')
        self.order_state = [0, 0.0]
        self.on_start_score = 10
        self.trades_made = [0, 0]

    def _load_data(self):
        df = pd.read_csv("../data/bitmex_1m_2016-2020.csv")
        df = df.drop(['symbol', 'timestamp'], axis=1)
        df.dropna(inplace=True)
        df['fMA'] = df.bidPrice.rolling(window=7).mean()
        df['sMA'] = df.bidPrice.rolling(window=30).mean()
        df['pct'] = df.bidPrice.pct_change()
        df['pctSum'] = df.pct.cumsum()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        if self.train_mode:
            data = df.iloc[-int((len(df) / 4) * 3):]
        else:
            data = df.iloc[:int((len(df) / 4))]
        data.reset_index(inplace=True)
        return data

    def reset(self):
        self.current_step = int(np.random.uniform(0, len(self.data) - (self.timeframe + self.window_size)))
        self.start = int(np.random.uniform(0, len(self.data) - (self.timeframe + self.window_size)))
        self.order_state = [0, 0.0]
        self.on_start_score = 10
        self.trades_made = [0, 0]
        observation = self._next_observation()
        return observation

    def _next_observation(self):
        data = self.data.iloc[
               self.current_step:self.current_step + self.window_size]  # bidSize, bidPrice, askPrice, askSize, fMA, sMA, pct, pctSum, =8
        ob = self._plot_window_to_np(data)
        self.current_data = data.iloc[-1]
        return ob

    def _plot_window_to_np(self, window_data):
        fig = plt.figure(figsize=(2, 2))  # must be equal
        plt.subplot(311)
        plt.plot(window_data.index, window_data[['bidPrice', 'askPrice', 'fMA', 'sMA']])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.subplot(312)
        plt.plot(window_data.index, window_data['pct'])
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.subplot(313)
        plt.plot(window_data.index, window_data['askSize'], 'rs', window_data.index, window_data['bidSize'], 'bs')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        fig.canvas.draw()
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='rgba')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        fig.clear()
        plt.close(fig)
        self.img_shape = [img_arr.shape[1], img_arr.shape[0], img_arr.shape[2]]
        # img = Image.fromarray(img_arr, 'RGBA')
        # img.show()
        return img_arr

    def step(self, action):
        prev_order_state = self.order_state[:]
        curr_action = self._take_action(action)
        reward = 0.0
        if prev_order_state[0] > self.order_state[0]:
            if prev_order_state[1] <= self.order_state[1]:
                reward += 1
                self.trades_made[0] += 1.0
            else:
                reward -= 1
                self.trades_made[1] += 1.0
            self.order_state[1] = 0.0
        self.on_start_score += reward
        done = self.current_step >= self.start + self.timeframe + self.window_size or self.on_start_score <= 0
        self.current_step += 1
        observation = self._next_observation()
        info = {}
        return observation, reward, done, info

    def _take_action(self, action):
        if action <= 1:
            if self.order_state[0] == 0:
                self.order_state[0] = 1
                self.order_state[1] = self.current_data['askPrice']
        elif action <= 2:
            if self.order_state[0] == 1:
                self.order_state[0] = 0
                self.order_state[1] = self.current_data['bidPrice']
        else:
            pass
        return action

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Score: {self.on_start_score}')
        print(f'Trades: {self.trades_made}')

    def close(self):
        pass


def random_agent(observation):
    action = [np.random.uniform(0, 3)]
    return action


env = DummyVecEnv([lambda: BitmexSimImg()])

model = PPO(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=500000)
model.save("PPO_BitmexSimImg")

# del model # remove to demonstrate saving and loading

# model = PPO.load("PPO_BitmexSimImg")

env2 = DummyVecEnv([lambda: BitmexSimImg(timeframe=330, train_mode=False)])
obs = env2.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    # action = random_agent(obs)
    env2.render()
    obs, reward, done, info = env2.step(action)
    if done:
      obs = env2.reset()


