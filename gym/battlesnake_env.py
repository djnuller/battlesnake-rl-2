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
        """Tag et trin i miljøet."""
        # Spilleren slanges handling
        direction = self._get_direction(action)
        new_head = {"x": self.snake[0]["x"] + direction["x"], "y": self.snake[0]["y"] + direction["y"]}

        # Tjek for kollision med vægge eller egen krop
        if not self._is_within_bounds(new_head) or new_head in self.snake:
            self.done = True
            step_data = {
                "you": {
                    "head": {"x": self.snake[0]["x"], "y": self.snake[0]["y"]},
                    "health": 0,  # Indikerer død
                    "body": self.snake,
                    "name": "PlayerSnake",
                },
                "turn": self.steps,
            }
            reward = self._calculate_reward(step_data)
            return self._get_observation(), reward, True, False, {}

        # Opdater spillerens slange
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self._generate_food()
        else:
            self.snake.pop()

        # Skab step_data for belønningsberegning
        step_data = {
            "you": {
                "head": new_head,
                "health": self.health - 1,  # Justér for hver tur
                "body": self.snake,
                "name": "PlayerSnake",
            },
            "board": {
                "food": [{"x": self.food["x"], "y": self.food["y"]}],
                "snakes": [
                    {"id": "OpponentSnake", "body": self.opponent.body},
                    {"id": "PlayerSnake", "body": self.snake},
                ],
            },
            "turn": self.steps,
        }

        # Tjek for vinder
        if self.steps >= 1000 or len(self.opponent.body) == 0:
            step_data["winnerName"] = "PlayerSnake"

        # Beregn belønning
        reward = self._calculate_reward(step_data)

        # Opdater modstanderen
        opponent_action = self.opponent.get_action(self._create_board())
        opponent_direction = self._get_direction(opponent_action)
        opponent_new_head = {
            "x": self.opponent.head["x"] + opponent_direction["x"],
            "y": self.opponent.head["y"] + opponent_direction["y"],
        }
        self.opponent.move(opponent_new_head)

        # Tjek for afslutning af spillet
        self.done = self.steps >= 1000
        self.steps += 1

        return self._get_observation(), reward, self.done, False, {}


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

    def _calculate_reward(self, step_data):
        reward = 0

        if step_data is not None:
            # Belønning for at spise mad
            if 'board' in step_data and 'food' in step_data['board']:
                food_positions = [(food['x'], food['y']) for food in step_data['board']['food']]
                if 'you' in step_data and 'head' in step_data['you']:
                    head_position = (step_data['you']['head']['x'], step_data['you']['head']['y'])
                    if head_position in food_positions:
                        reward += 15  # Øget belønning for at spise mad

            # Straf for at dø
            if 'you' in step_data and 'health' in step_data['you']:
                if step_data['you']['health'] <= 0:
                    reward -= 200  # Forhøjet straf for at dø

            # Belønning for at vinde
            if 'winnerName' in step_data and 'you' in step_data and 'name' in step_data['you']:
                if step_data['winnerName'] == step_data['you']['name']:
                    reward += 500  # Forstør belønning for at vinde spillet

            # Belønning for overlevelse
            if 'turn' in step_data and step_data['turn'] > 0:
                survival_reward = 0.5 + (step_data['turn'] * 0.01)
                reward += survival_reward

            # Straf for farlige positioner
            if 'you' in step_data and 'head' in step_data['you'] and 'board' in step_data and 'snakes' in step_data['board']:
                head_position = (step_data['you']['head']['x'], step_data['you']['head']['y'])
                snake_bodies = [part for snake in step_data['board']['snakes'] if 'body' in snake for part in snake['body'] if snake.get('id') != step_data['you'].get('id')]
                own_body = step_data['you']['body'][1:]  # Undgå hovedet
                danger_positions = [(part['x'], part['y']) for part in snake_bodies + own_body]
                if head_position not in danger_positions:
                    reward += 5
                else:
                    reward -= 20  # Øget straf for farer

            # Belønning for at reducere modstanderes muligheder
            if 'board' in step_data and 'snakes' in step_data['board']:
                for snake in step_data['board']['snakes']:
                    if 'id' in snake and snake['id'] != step_data['you'].get('id', None):
                        if 'body' in snake and len(snake['body']) < len(step_data['you'].get('body', [])):
                            reward += 10  # Øget belønning for dominans

            # Strategisk positionering (forblive i centrum af brættet)
            if 'you' in step_data and 'head' in step_data['you'] and 'turn' in step_data:
                head_x, head_y = step_data['you']['head']['x'], step_data['you']['head']['y']
                center_x, center_y = self.width // 2, self.height // 2  # Brug brættets dimensioner
                distance_from_center = abs(head_x - center_x) + abs(head_y - center_y)
                early_game_bonus = 15 if step_data['turn'] < 50 else 10
                position_reward = max(0, early_game_bonus - distance_from_center)
                reward += position_reward


        return reward
