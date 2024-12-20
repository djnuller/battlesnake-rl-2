import os
from re import VERBOSE
from stable_baselines3 import PPO
from battlesnake_env import BattlesnakeEnv

# Konfigurationsparametre
MODELS_DIR = "models"
EVALUATION_GAMES = 100  # Reduceret antal evalueringer
TRAINING_TIMESTEPS = 100000
GENERATIONS = 100  # Antal iterationer i træningscyklussen

# Sikre, at models-mappen findes
os.makedirs(MODELS_DIR, exist_ok=True)

def get_latest_model():
    """Find the latest generation model in the models directory."""
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
    if not models:
        return None

    # Sort models by generation and model number
    models.sort(key=lambda x: (int(x.split("_gen")[-1].split("_")[0]), int(x.split("_")[-1].split(".")[0])))
    return os.path.join(MODELS_DIR, models[-1])

def evaluate_all_models():
    """Evaluate all models and find the best one."""
    models = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
    if len(models) < 3:
        return None

    scores = {}
    for model_path in models:
        scores[model_path] = evaluate_model(model_path, EVALUATION_GAMES)

    best_model_path = max(scores, key=scores.get)
    print(f"Bedste eksisterende model: {best_model_path} med score: {scores[best_model_path]}")
    return best_model_path

def train_model(base_model_path, new_model_path, timesteps):
    """Træn en ny model baseret på en eksisterende."""
    # Indlæs basemodellen eller opret en ny
    if base_model_path:
        model = PPO.load(base_model_path, device='cpu', learning_rate=0.0003, ent_coef=0.005)
    else:
        env = BattlesnakeEnv()
        model = PPO("MlpPolicy", env, verbose=1, device='cpu', learning_rate=0.0003, ent_coef=0.005)

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
    # Start med en specificeret base model, eller find den seneste model
    base_model_path = start_model or get_latest_model()

    # Hvis der er mere end 3 modeller, evaluér for at finde den bedste
    if not start_model:
        base_model_path = evaluate_all_models() or base_model_path

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
    main()
