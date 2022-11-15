from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.env_checker import check_env
from bitmex_limited import BitmexLimited
import numpy as np

env = BitmexLimited(episode_len=1, train_mode=False)
check_env(env, warn=True)

obs = env.reset()

n_steps = 90
for x in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if x == 89:
        env.render()
    if done:
        print('Done')
        env.render()
        obs = env.reset()
