from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from battlesnake_env import BattlesnakeEnv

env = make_vec_env(BattlesnakeEnv, n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, device="cpu")

model.learn(total_timesteps=100000)

model.save("trained_model.zip")
