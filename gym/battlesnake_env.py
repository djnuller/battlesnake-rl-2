import gymnasium
from gymnasium import spaces
import numpy as np
import random
from simple_snake import SimpleSnake


class BattlesnakeEnv(gymnasium.Env):
    def __init__(self, width=11, height=11):
        super(BattlesnakeEnv, self).__init__()
        self.width = width
        self.height = height

        # Observation space: the board as a flat array
        self.observation_space = spaces.Box(low=0, high=3, shape=(width * height,), dtype=np.int32)

        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Modstander slange
        self.opponent = SimpleSnake()

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Set the seed if provided
        if seed is not None:
            self.seed(seed)

        self.snake = [{"x": self.width // 2, "y": self.height // 2}]
        self.food = self._generate_food()
        self.done = False
        self.steps = 0
        self.health = 100

        # Reset modstanderslangen
        self.opponent.reset(self.width, self.height)

        return self._get_observation(), {}


    def step(self, action):
        # Spilleren slanges handling
        direction = self._get_direction(action)
        new_head = {"x": self.snake[0]["x"] + direction["x"], "y": self.snake[0]["y"] + direction["y"]}

        # Tjek for kollision med vægge eller egen krop
        if not self._is_within_bounds(new_head) or new_head in self.snake:
            self.done = True
            return self._get_observation(), -100, True, False, {}

        # Opdater spillerens slange
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10
            self.food = self._generate_food()
        else:
            reward = 1
            self.snake.pop()

        # Opret board
        board = self._create_board()

        # Modstander slanges handling
        opponent_action = self.opponent.get_action(board)
        opponent_direction = self._get_direction(opponent_action)
        opponent_new_head = {
            "x": self.opponent.head["x"] + opponent_direction["x"],
            "y": self.opponent.head["y"] + opponent_direction["y"],
        }

        # Tjek for kollision mellem spilleren og modstanderen
        if new_head == opponent_new_head:
            if len(self.snake) > len(self.opponent.body):
                reward += 500  # Spilleren vinder kollisionen
                self.done = True
            else:
                reward = -100  # Spilleren taber kollisionen
                self.done = True
            return self._get_observation(), reward, True, False, {}

        # Opdater modstanderen
        self.opponent.move(opponent_new_head)

        # Tjek for kollision med modstanderen
        if new_head in self.opponent.body:
            reward = -100
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Tjek om modstanderen rammer spilleren
        if opponent_new_head in self.snake:
            self.done = True
            return self._get_observation(), -100, True, False, {}

        # Truncation og returnering
        truncated = self.steps >= 1000
        self.steps += 1

        return self._get_observation(), reward, self.done, truncated, {}


    def _get_observation(self):
        board = np.zeros((self.width, self.height), dtype=np.int32)

        # Spilleren slange
        board[self.snake[0]["y"], self.snake[0]["x"]] = 1
        for segment in self.snake[1:]:
            board[segment["y"], segment["x"]] = 2

        # Modstander slange
        board[self.opponent.head["y"], self.opponent.head["x"]] = 3
        for segment in self.opponent.body[1:]:
            board[segment["y"], segment["x"]] = 4

        # Mad
        board[self.food["y"], self.food["x"]] = 5

        return board.flatten()


    def _generate_food(self):
        while True:
            food = {"x": random.randint(0, self.width - 1), "y": random.randint(0, self.height - 1)}
            if food not in self.snake and food not in self.opponent.body:
                return food

    def _is_within_bounds(self, position):
        return 0 <= position["x"] < self.width and 0 <= position["y"] < self.height

    def _get_direction(self, action):
        """Map action to direction."""
        directions = {
            "up": {"x": 0, "y": -1},
            "down": {"x": 0, "y": 1},
            "left": {"x": -1, "y": 0},
            "right": {"x": 1, "y": 0}
        }

        if isinstance(action, str):
            return directions[action]  # Handle string inputs
        else:
            # For numerical actions (e.g., from RL agents)
            index_to_str = ["up", "down", "left", "right"]
            return directions[index_to_str[action]]

    def _create_board(self):
        """Create a board representation for the environment."""
        board = np.zeros((self.width, self.height), dtype=np.int32)

        # Spilleren slange
        board[self.snake[0]["y"], self.snake[0]["x"]] = 1  # Head
        for segment in self.snake[1:]:
            board[segment["y"], segment["x"]] = 2  # Body

        # Modstander slange
        board[self.opponent.head["y"], self.opponent.head["x"]] = 3  # Head
        for segment in self.opponent.body[1:]:
            board[segment["y"], segment["x"]] = 4  # Body

        # Mad
        board[self.food["y"], self.food["x"]] = 5

        return board


    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]
