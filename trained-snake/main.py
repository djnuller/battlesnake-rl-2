from flask import Flask, request, jsonify
from stable_baselines3 import PPO
import numpy as np

app = Flask(__name__)

# Indlæs den trænede model
MODEL_PATH = "model_gen40_2.zip"
model = PPO.load(MODEL_PATH)

# Funktion til at konvertere Battlesnake API-data til observationsformat
def create_observation(data):
    board = data["board"]
    width, height = board["width"], board["height"]

    observation = np.zeros((height, width), dtype=np.int32)

    # Placér din slange
    you = data["you"]

    # Placér modstandere
    for snake in board["snakes"]:
        if snake["id"] != you["id"]:
            for segment in snake["body"]:
                observation[segment["y"], segment["x"]] = 3  # Krop
            observation[snake["head"]["y"], snake["head"]["x"]] = 4  # Hoved


    for segment in you["body"]:
        observation[segment["y"], segment["x"]] = 1  # Krop
    observation[you["head"]["y"], you["head"]["x"]] = 2  # Hoved



    # Placér mad
    for food in board["food"]:
        observation[food["y"], food["x"]] = 5

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

    return jsonify({"move": direction})

@app.route("/end", methods=["POST"])
def end():
    print("Game ended!")
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
