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

        self.snake = [
            {"x": self.width // 2, "y": self.height // 2},
            {"x": self.width // 2, "y": self.height // 2 + 1},
            {"x": self.width // 2, "y": self.height // 2 + 2}
        ]
        self.food = self._generate_food()
        self.done = False
        self.steps = 0
        self.health = 100

        # Reset modstanderslangen
        self.opponent.reset(self.width, self.height)
        self.opponent.body = [
            {"x": self.width // 4, "y": self.height // 4},
            {"x": self.width // 4, "y": self.height // 4 + 1},
            {"x": self.width // 4, "y": self.height // 4 + 2}
        ]

        return self._get_observation(), {}

    def step(self, action):
        """Tag et trin i miljøet."""
        direction = self._get_direction(action)
        new_head = {"x": self.snake[0]["x"] + direction["x"], "y": self.snake[0]["y"] + direction["y"]}

        # Modstanderen foretager et træk
        opponent_board = self._get_observation().reshape(self.height, self.width)  # Sørg for 2D-format
        opponent_action = self.opponent.get_action(opponent_board)
        opponent_direction = self._get_direction(opponent_action)
        opponent_new_head = {
            "x": self.opponent.head["x"] + opponent_direction["x"],
            "y": self.opponent.head["y"] + opponent_direction["y"],
        }

        player_collision = (
            not self._is_within_bounds(new_head) or
            new_head in self.snake or
            new_head in self.opponent.body
        )

        opponent_collision = (
            not self._is_within_bounds(opponent_new_head) or
            opponent_new_head in self.opponent.body or
            opponent_new_head in self.snake
        )

        step_data = {
            "you": {
                "id": "PlayerSnake",
                "head": new_head,
                "health": self.health,
                "body": self.snake,
                "name": "PlayerSnake"
            },
            "board": {
                "width": self.width,
                "height": self.height,
                "food": [{"x": self.food["x"], "y": self.food["y"]}],
                "snakes": [
                    {"id": "PlayerSnake", "body": self.snake, "head": new_head},
                    {"id": "OpponentSnake", "body": self.opponent.body, "head": opponent_new_head}
                ]
            },
            "turn": self.steps
        }

        if player_collision:
            #self.done = True
            step_data["you"]["health"] = 0
            reward = self._calculate_reward(step_data)
            return self._get_observation(), reward, self.done, False, {}

        if opponent_collision:
            #self.done = True
            step_data["winnerName"] = "PlayerSnake"
            reward = self._calculate_reward(step_data)
            return self._get_observation(), reward, self.done, False, {}

        if new_head == self.food:
            reward = 100  # Belønning for mad
            self.health = min(100, self.health + 20)  # Øg sundhed med 20, men maksimer ved 100
            self.food = self._generate_food()
        else:
            reward = -1  # Straf for ikke at spise mad
            self.snake.pop()

        self.snake.insert(0, new_head)  # Slangen vokser altid ved at indsætte nyt hoved

        self.opponent.move(opponent_new_head)

        self.health -= 1
        self.steps += 1

        if self.health <= 0:
            self.done = True

        reward += self._calculate_reward(step_data)

        return self._get_observation(), reward, self.done, False, {}

    def _get_observation(self):
        board = np.zeros((self.width, self.height), dtype=np.int32)

        board[self.snake[0]["y"], self.snake[0]["x"]] = 1
        for segment in self.snake[1:]:
            board[segment["y"], segment["x"]] = 2

        board[self.opponent.head["y"], self.opponent.head["x"]] = 3
        for segment in self.opponent.body[1:]:
            board[segment["y"], segment["x"]] = 4

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
        directions = {
            "up": {"x": 0, "y": -1},
            "down": {"x": 0, "y": 1},
            "left": {"x": -1, "y": 0},
            "right": {"x": 1, "y": 0}
        }

        if isinstance(action, str):
            return directions[action]
        else:
            index_to_str = ["up", "down", "left", "right"]
            return directions[index_to_str[action]]

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _calculate_reward(self, step_data):
        reward = 0

        # Belønning for at spise mad
        reward += self.reward_for_food(step_data)

        # Straf for kollisioner
        reward += self.penalty_for_collisions(step_data)

        # Belønning for overlevelse
        reward += 1

        return reward

    def reward_for_food(self, step_data):
        reward = 0
        if 'board' in step_data and 'food' in step_data['board']:
            food_positions = [(food['x'], food['y']) for food in step_data['board']['food']]
            if 'you' in step_data and 'head' in step_data['you']:
                head_position = (step_data['you']['head']['x'], step_data['you']['head']['y'])
                if head_position in food_positions:
                    reward += 100  # Belønning for at spise mad
        return reward

    def penalty_for_collisions(self, step_data):
        penalty = 0
        if 'you' in step_data and 'head' in step_data['you']:
            head_position = (step_data['you']['head']['x'], step_data['you']['head']['y'])

            if (head_position[0] < 0 or head_position[0] >= self.width or
                    head_position[1] < 0 or head_position[1] >= self.height):
                penalty -= 500  # Høj straf for at ramme væggen

            elif head_position in [(segment['x'], segment['y']) for segment in step_data['you']['body'][1:]]:
                penalty -= 500  # Høj straf for at ramme egen krop

            for snake in step_data['board']['snakes']:
                if snake['id'] != step_data['you']['id']:
                    opponent_positions = [(segment['x'], segment['y']) for segment in snake['body']]
                    if head_position in opponent_positions:
                        penalty -= 500  # Høj straf for at ramme modstanderens krop

        return penalty

    def _get_safe_moves(self, head_position):
        """Return a list of safe moves based on the current head position."""
        safe_moves = []
        directions = {
            "up": {"x": 0, "y": -1},
            "down": {"x": 0, "y": 1},
            "left": {"x": -1, "y": 0},
            "right": {"x": 1, "y": 0}
        }

        for move, delta in directions.items():
            new_position = {"x": head_position["x"] + delta["x"], "y": head_position["y"] + delta["y"]}
            if self._is_within_bounds(new_position) and \
               new_position not in self.snake and \
               new_position not in self.opponent.body:
                safe_moves.append(move)

        return safe_moves
