from flask import Flask, request, jsonify
import random
from collections import deque

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return {
      "apiversion": "1",
      "author": "MyUsername",
      "color": "#888888",
      "head": "beluga",
      "tail": "round-bum",
      "version": "0.0.1-beta"
    }

@app.route("/start", methods=["POST"])
def start():
    return jsonify({"color": "#88CC88", "headType": "beluga", "tailType": "round-bum"})

@app.route("/move", methods=["POST"])
def move():
    data = request.json

    # Board and snake details
    board = data["board"]
    width, height = board["width"], board["height"]
    you = data["you"]
    head = you["head"]
    body = you["body"]
    health = you["health"]
    length = you["length"]

    # Other snakes
    snakes = board["snakes"]
    max_opponent_length = max(snake["length"] for snake in snakes if snake["id"] != you["id"])

    # Food locations
    food = board["food"]

    # Directions
    moves = {
        "up": {"x": head["x"], "y": head["y"] + 1},
        "down": {"x": head["x"], "y": head["y"] - 1},
        "left": {"x": head["x"] - 1, "y": head["y"]},
        "right": {"x": head["x"] + 1, "y": head["y"]}
    }

    # Check if a cell is safe
    def is_safe(cell):
        # Check board boundaries
        if cell["x"] < 0 or cell["x"] >= width or cell["y"] < 0 or cell["y"] >= height:
            return False
        # Check collisions with any snake body
        for snake in snakes:
            if {"x": cell["x"], "y": cell["y"]} in snake["body"]:
                return False
        return True

    # Evaluate threats from opponent heads
    def is_threatened(cell):
        for snake in snakes:
            if snake["id"] != you["id"] and snake["length"] >= length:
                for move in moves.values():
                    next_head = {"x": snake["head"]["x"] + move["x"] - head["x"], "y": snake["head"]["y"] + move["y"] - head["y"]}
                    if next_head == cell:
                        return True
        return False

    # Flood-fill to estimate space
    def flood_fill(start):
        visited = set()
        queue = deque([start])
        area = 0
        while queue:
            cell = queue.popleft()
            if (cell["x"], cell["y"]) in visited:
                continue
            visited.add((cell["x"], cell["y"]))
            area += 1
            for direction in moves.values():
                neighbor = {"x": cell["x"] + direction["x"] - head["x"], "y": cell["y"] + direction["y"] - head["y"]}
                if is_safe(neighbor) and (neighbor["x"], neighbor["y"]) not in visited:
                    queue.append(neighbor)
        return area

    # Safe moves
    safe_moves = {move: pos for move, pos in moves.items() if is_safe(pos) and not is_threatened(pos)}

    # Find food only if necessary and uncontested
    if food and (length < max_opponent_length or health <= 80):
        best_food = None
        shortest_distance = float("inf")

        for f in food:
            # Calculate distance to food for this snake
            my_distance = abs(head["x"] - f["x"]) + abs(head["y"] - f["y"])

            # Check if any opponent's head is as close or closer to this food
            contested = False
            for snake in snakes:
                if snake["id"] != you["id"]:
                    opponent_distance = abs(snake["head"]["x"] - f["x"]) + abs(snake["head"]["y"] - f["y"])
                    if opponent_distance <= my_distance:  # Contested or opponent is closer
                        contested = True
                        break

            # Prioritize food that is uncontested and closest
            if not contested and my_distance < shortest_distance:
                best_food = f
                shortest_distance = my_distance

        # Use BFS to move towards the selected food if it's safe
        if best_food:
            visited = set()
            queue = deque([(head, [])])  # (current position, path to get there)
            while queue:
                current, path = queue.popleft()
                if (current["x"], current["y"]) in visited:
                    continue
                visited.add((current["x"], current["y"]))

                # Check if we've reached the best food
                if current == best_food:
                    return jsonify({"move": path[0]})  # Return the first step in the path

                # Add neighbors to the queue
                for move, direction in moves.items():
                    neighbor = {"x": current["x"] + direction["x"] - head["x"], "y": current["y"] + direction["y"] - head["y"]}
                    if is_safe(neighbor):
                        queue.append((neighbor, path + [move]))

    # Flood-fill for each safe move
    move_scores = {move: flood_fill(pos) for move, pos in safe_moves.items()}

    # Choose move with maximum space
    if move_scores:
        best_move = max(move_scores, key=move_scores.get)
    else:
        best_move = random.choice(list(safe_moves.keys())) if safe_moves else "up"

    return jsonify({"move": best_move})



@app.route("/end", methods=["POST"])
def end():
    return "Game over", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
