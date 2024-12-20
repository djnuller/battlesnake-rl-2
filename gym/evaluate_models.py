from stable_baselines3 import PPO
from battlesnake_env import BattlesnakeEnv

models = ["model1.zip", "model2.zip", "model3.zip", "model4.zip"]

scores = {}

for model_path in models:
    print(f"Evaluating model: {model_path}")
    model = PPO.load(model_path)
    score = 0

    env = BattlesnakeEnv()

    obs, _ = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, _ = env.step(action)
        score += reward

        if done or truncated:
            obs, _ = env.reset()

    scores[model_path] = score

best_model = max(scores, key=scores.get)
print(f"Best model: {best_model} with score: {scores[best_model]}")
