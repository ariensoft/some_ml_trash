import gym
import numpy as np
from envs.bitmex_limited import BitmexLimited
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: BitmexLimited()])

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log='../logs/td3_indicator/')
model.learn(total_timesteps=600000, log_interval=10)
model.save("../models/td3_indicator")
env = model.get_env()

del model  # remove to demonstrate saving and loading

model = TD3.load("../models/td3_indicator")

env2 = DummyVecEnv([lambda: BitmexLimited(train_mode=False)])
obs = env2.reset()
n_steps = 1440*100
for x in range(n_steps):
    action, _states = model.predict(obs)
    obs, reward, done, info = env2.step(action)
    if x == n_steps - 1:
        env2.render()
    if done:
        print('Done', info[0]['terminal_reason'])
        obs = env2.reset()


