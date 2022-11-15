from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from envs.bitmex_limited import BitmexLimited
from envs.stenv import BitstampEnv
import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)

env = DummyVecEnv([lambda: BitstampEnv()])

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log='../logs/ppo/')
model.learn(total_timesteps=400000)
model.save("../models/PPO")

del model  # remove to demonstrate saving and loading

model = PPO.load("../models/PPO")

env2 = DummyVecEnv([lambda: BitstampEnv(train_mode=False)])
obs = env2.reset()

done = False
while not done:
    # Random action
    #action = env.action_space.sample()
    action, _states = model.predict(obs)
    obs, reward, done, info = env2.step(action)
    env2.render()
    if done:
        print('Done', info)

