from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from envs.bitmex_limited import BitmexLimited
import numpy as np

env = DummyVecEnv([lambda: BitmexLimited()])

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log='../logs/ppo/')
model.learn(total_timesteps=675000)
model.save("../models/DDPG")

del model  # remove to demonstrate saving and loading

model = DDPG.load("../models/DDPG")

env2 = DummyVecEnv([lambda: BitmexLimited(train_mode=False)])
obs = env2.reset()

n_steps = 1440*100
for x in range(n_steps):
    # Random action
    #action = env.action_space.sample()
    action, _states = model.predict(obs)
    obs, reward, done, info = env2.step(action)
    if x == n_steps - 1:
        env2.render()
    if done:
        print('Done', info[0]['terminal_reason'])
        obs = env2.reset()
