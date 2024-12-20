from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from battlesnake_env import BattlesnakeEnv

# Opret miljø
env = make_vec_env(BattlesnakeEnv, n_envs=1)

# Opret model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # Skift til "MlpPolicy", hvis du vil

# Træn modellen
model.learn(total_timesteps=100000)

# Gem modellen
model.save("trained_model.zip")
