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

        # Tjek for kollisioner (spilleren)
        player_collision = (
            not self._is_within_bounds(new_head) or  # Væg
            new_head in self.snake or              # Egen krop
            new_head in self.opponent.body         # Modstanderens krop
        )

        # Tjek for kollisioner (modstanderen)
        opponent_collision = (
            not self._is_within_bounds(opponent_new_head) or  # Væg
            opponent_new_head in self.opponent.body or       # Egen krop
            opponent_new_head in self.snake                  # Spillerens krop
        )

        # Opret step_data til belønningsberegning
        step_data = {
            "you": {
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

        # Håndter kollisioner
        if player_collision:
            self.done = True
            step_data["you"]["health"] = 0  # Spiller død
            reward = self._calculate_reward(step_data)
            return self._get_observation(), reward, self.done, False, {}

        if opponent_collision:
            self.done = True
            step_data["winnerName"] = "PlayerSnake"  # Spiller vinder
            reward = self._calculate_reward(step_data)
            return self._get_observation(), reward, self.done, False, {}

        # Opdater spillerens slange
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10  # Belønning for mad
            self.food = self._generate_food()
        else:
            reward = 1  # Belønning for at overleve
            self.snake.pop()

        # Opdater modstanderen
        self.opponent.move(opponent_new_head)

        # Opdater antal træk og sundhed
        self.health -= 1
        self.steps += 1

        # Slut spillet kun efter et maksimalt antal træk
        if self.steps >= 100 or self.health <= 0:
            self.done = True

        # Beregn reward fra `_calculate_reward`
        reward += self._calculate_reward(step_data)

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
                        reward += 50  # Stor belønning for at spise mad

            # Straf for at dø
            if 'you' in step_data and 'health' in step_data['you']:
                if step_data['you']['health'] <= 0:
                    reward -= 1000  # Hårdere straf for at dø

            # Belønning for at vinde
            if 'winnerName' in step_data and 'you' in step_data and 'name' in step_data['you']:
                if step_data['winnerName'] == step_data['you']['name']:
                    reward += 2000  # Stor belønning for at vinde spillet

            # Belønning for overlevelse
            if 'turn' in step_data and step_data['turn'] > 0:
                reward += 1  # Konstant belønning for at overleve

            # Straf for farlige positioner
            if 'you' in step_data and 'head' in step_data['you'] and 'board' in step_data and 'snakes' in step_data['board']:
                head_position = (step_data['you']['head']['x'], step_data['you']['head']['y'])
                snake_bodies = [part for snake in step_data['board']['snakes'] if 'body' in snake for part in snake['body'] if snake.get('id') != step_data['you'].get('id')]
                own_body = step_data['you']['body'][1:]  # Undgå hovedet
                danger_positions = [(part['x'], part['y']) for part in snake_bodies + own_body]

                # Straf for at være tæt på farer
                danger_radius = 1  # Antal felter fra hovedet, der regnes som farlige
                close_dangers = sum(
                    1 for danger in danger_positions
                    if abs(danger[0] - head_position[0]) <= danger_radius and abs(danger[1] - head_position[1]) <= danger_radius
                )
                reward -= close_dangers * 10  # Straf for hver fare i nærheden

                # Straf for faktisk at ramme farer
                if head_position in danger_positions:
                    reward -= 500  # Stor straf for at ramme fare

            # Straf for at ramme vægge
            head_position = (step_data['you']['head']['x'], step_data['you']['head']['y'])
            if (head_position[0] < 0 or head_position[0] >= self.width or
                    head_position[1] < 0 or head_position[1] >= self.height):
                reward -= 1000  # Hårdere straf for at ramme væggen

            # Straf for at ramme egen krop
            elif head_position in [(segment['x'], segment['y']) for segment in step_data['you']['body'][1:]]:
                reward -= 1000  # Hårdere straf for at ramme egen krop

            # Belønning for at holde afstand fra vægge
            distance_to_wall = min(
                head_position[0],  # Afstand til venstre væg
                self.width - head_position[0] - 1,  # Afstand til højre væg
                head_position[1],  # Afstand til top væg
                self.height - head_position[1] - 1  # Afstand til bund væg
            )
            reward += distance_to_wall * 2  # Belønning for at holde sig væk fra vægge

        return reward
