import random
from collections import deque
import numpy as np

class SimpleSnake:
    def __init__(self):
        """Initialize the snake with default values."""
        self.head = None
        self.body = []

    def reset(self, width, height):
        """Reset the snake to its initial position."""
        self.head = {"x": width // 4, "y": height // 4}
        self.body = [self.head]

    def _heuristic_space(self, board, start):
            """Estimer plads baseret på Manhattan-afstand."""
            width, height = board.shape
            x, y = start["x"], start["y"]

            # Tæl gyldige celler inden for en bestemt afstand
            max_distance = min(width, height)  # Begræns søgeområdet
            score = 0

            for dx in range(-max_distance, max_distance + 1):
                for dy in range(-max_distance, max_distance + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and abs(dx) + abs(dy) <= max_distance:
                        if board[ny, nx] == 0:  # Kun frie celler
                            score += 1

            return score

    def get_action(self, board):
        """Determine the action for the snake based on the board."""
        width, height = board.shape
        moves = {
            "up": {"x": 0, "y": -1},
            "down": {"x": 0, "y": 1},
            "left": {"x": -1, "y": 0},
            "right": {"x": 1, "y": 0},
        }

        valid_moves = {}

        # Filtrer gyldige træk
        for move, delta in moves.items():
            next_head = {"x": self.head["x"] + delta["x"], "y": self.head["y"] + delta["y"]}
            if 0 <= next_head["x"] < width and 0 <= next_head["y"] < height:
                cell_value = board[next_head["y"], next_head["x"]]
                if cell_value in (0, 5):  # Fri plads eller mad
                    valid_moves[move] = next_head

        # Evaluér gyldige træk med _heuristic_space
        if valid_moves:
            move_scores = {move: self._heuristic_space(board, next_head) for move, next_head in valid_moves.items()}
            # Prioritér mad, hvis muligt
            for move, next_head in valid_moves.items():
                if board[next_head["y"], next_head["x"]] == 5:  # Mad
                    return move
            # Vælg det bedste træk baseret på plads
            return max(move_scores, key=move_scores.get)

        # Fallback: Tilfældig handling
        return random.choice(list(moves.keys()))


    def move(self, new_head):
        """Update the snake's position based on the new head."""
        self.body.insert(0, new_head)  # Add new head to the body
        self.head = new_head  # Update the head reference
        self.body.pop()  # Remove the tail segment to simulate movement
