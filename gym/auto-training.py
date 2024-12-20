import os
from stable_baselines3 import PPO
from battlesnake_env import BattlesnakeEnv

# Konfigurationsparametre
MODELS_DIR = "models"
EVALUATION_GAMES = 200  # Reduceret antal evalueringer
TRAINING_TIMESTEPS = 100000
GENERATIONS = 100  # Antal iterationer i træningscyklussen

# Sikre, at models-mappen findes
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(base_model_path, new_model_path, timesteps):
    """Træn en ny model baseret på en eksisterende."""
    # Indlæs basemodellen eller opret en ny
    if base_model_path:
        model = PPO.load(base_model_path, device='cpu')
    else:
        env = BattlesnakeEnv()
        model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    # Initialiser miljø
    env = BattlesnakeEnv()
    model.set_env(env)

    # Træn modellen
    model.learn(total_timesteps=timesteps)

    # Gem den trænede model
    model.save(new_model_path)
    print(f"Model gemt: {new_model_path}")

def evaluate_model(model_path, games):
    """Evaluér en model over et antal spil."""
    model = PPO.load(model_path, device='cpu')
    env = BattlesnakeEnv()

    total_score = 0
    for _ in range(games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_score += reward
            if done or truncated:
                break

    avg_score = total_score / games
    print(f"Model {model_path} gennemsnitlig score: {avg_score}")
    return avg_score

def main(start_model=None):
    # Start med en specificeret base model, hvis angivet
    base_model_path = start_model

    for generation in range(1, GENERATIONS + 1):
        print(f"=== Generation {generation} ===")
        new_models = []

        # Træn 3 nye modeller
        for i in range(3):
            new_model_path = os.path.join(MODELS_DIR, f"model_gen{generation}_{i + 1}.zip")
            train_model(base_model_path, new_model_path, TRAINING_TIMESTEPS)
            new_models.append(new_model_path)

        # Evaluér alle modeller
        scores = {}
        for model_path in new_models:
            scores[model_path] = evaluate_model(model_path, EVALUATION_GAMES)

        # Vælg den bedste model
        best_model_path = max(scores, key=scores.get)
        print(f"Bedste model: {best_model_path} med score: {scores[best_model_path]}")

        # Brug den bedste model som base for næste generation
        base_model_path = best_model_path

if __name__ == "__main__":
    # Start fra model_gen1_3.zip og juster evalueringskampe
    main(start_model=os.path.join(MODELS_DIR, "trained_model.zip"))
