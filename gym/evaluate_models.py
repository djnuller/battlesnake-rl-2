from stable_baselines3 import PPO
from battlesnake_env import BattlesnakeEnv

# Liste over modeller
models = ["model1.zip", "model2.zip", "model3.zip", "model4.zip"]

# Score-holder
scores = {}

# Evaluér hver model
for model_path in models:
    print(f"Evaluating model: {model_path}")
    model = PPO.load(model_path)
    score = 0

    # Initialiser miljøet
    env = BattlesnakeEnv()

    # Reset miljøet
    obs, _ = env.reset()

    # Spil 100 episoder
    for _ in range(1000):
        # Få handling fra modellen
        action, _ = model.predict(obs, deterministic=True)

        # Udfør handling
        obs, reward, done, truncated, _ = env.step(action)
        score += reward  # Tilføj belønning til score

        # Reset, hvis spillet slutter
        if done or truncated:
            obs, _ = env.reset()

    # Gem scoren for modellen
    scores[model_path] = score

# Find den bedste model
best_model = max(scores, key=scores.get)
print(f"Best model: {best_model} with score: {scores[best_model]}")
