from envs.bitmex_limited import BitmexLimited
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy, CnnPolicy

env = DummyVecEnv([lambda: BitmexLimited()])

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='../logs/ppo/')
model.learn(total_timesteps=1000000)
model.save("../models/DQN")

del model  # remove to demonstrate saving and loading

model = DQN.load("../models/DQN")

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
