from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from envs.bitmex_simulated import BitmexSimulated
from envs.bitmex_simple import BitmexContinuous


env = DummyVecEnv([lambda: BitmexContinuous()])

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log='../logs/ppo/')
model.learn(total_timesteps=1000000)
model.save("../models/A2C")

del model  # remove to demonstrate saving and loading

model = A2C.load("../models/A2C")

env2 = DummyVecEnv([lambda: BitmexContinuous(train_mode=False)])
obs = env2.reset()
done = False
c = 0
while not done:
    action, _states = model.predict(obs)
    # action = random_agent(obs)
    env2.render()
    obs, reward, done, info = env2.step(action)
    c += 1
    print(c, done)