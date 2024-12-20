from flask import Flask, request, jsonify
from stable_baselines3 import PPO
import numpy as np

app = Flask(__name__)

# Indlæs den trænede model
MODEL_PATH = "model_gen1_1.zip"
model = PPO.load(MODEL_PATH)

# Funktion til at konvertere Battlesnake API-data til observationsformat
def create_observation(data):
    board = data["board"]
    width, height = board["width"], board["height"]

    # Spejl y-koordinaten
    def flip_y(y):
        return height - 1 - y

    observation = np.zeros((height, width), dtype=np.int32)

    # Placér modstandere
    for snake in board["snakes"]:
        for segment in snake["body"]:
            observation[flip_y(segment["y"]), segment["x"]] = 4  # Krop
        observation[flip_y(snake["head"]["y"]), snake["head"]["x"]] = 3  # Hoved

    # Placér din slange
    you = data["you"]
    for segment in you["body"]:
        observation[flip_y(segment["y"]), segment["x"]] = 2  # Krop
    observation[flip_y(you["head"]["y"]), you["head"]["x"]] = 1  # Hoved

    # Placér mad
    for food in board["food"]:
        observation[flip_y(food["y"]), food["x"]] = 5

    print(observation)

    return observation.flatten()


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "apiversion": "1",
        "author": "your_name",
        "color": "#888888",
        "head": "default",
        "tail": "default"
    })

@app.route("/start", methods=["POST"])
def start():
    print("Game started!")
    return "OK"

@app.route("/move", methods=["POST"])
def move():
    data = request.json

    # Opret observation fra data
    observation = create_observation(data)

    # Brug modellen til at forudsige næste træk
    action, _states = model.predict(observation)

    # Konverter numerisk handling til retning
    actions = ["up", "down", "left", "right"]
    direction = actions[action]
    print(direction)

    return jsonify({"move": direction})

@app.route("/end", methods=["POST"])
def end():
    print("Game ended!")
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
