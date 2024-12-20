from stable_baselines3 import PPO
from battlesnake_env import BattlesnakeEnv

best_model_path = "model4.zip"

model = PPO.load(best_model_path, device="cpu")

env = BattlesnakeEnv()

model.set_env(env)
model.learn(total_timesteps=100000)
model.save("best_model_updated.zip")

print("Training complete. Updated model saved as 'best_model_updated.zip'.")
