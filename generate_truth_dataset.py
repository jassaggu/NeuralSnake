import pygame
import numpy as np

# -------------------
# Config
# -------------------
GRID_SIZE = 10
CELL_SIZE = 20
CHANNELS = 3
DATASET_SIZE = 50000
OUTPUT_FILE = "snake_transitions.npz"

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

pygame.init()


# -------------------
# Snake Game
# -------------------
class SnakeGame:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = RIGHT
        self.food = self.spawn_food()
        self.done = False

    def spawn_food(self):
        while True:
            pos = (np.random.randint(self.grid_size),
                   np.random.randint(self.grid_size))
            if pos not in self.snake:
                return pos

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size, CHANNELS), dtype=np.float32)

        for x, y in self.snake[:-1]:
            state[y, x, 0] = 1  # body

        head_x, head_y = self.snake[-1]
        state[head_y, head_x, 1] = 1  # head

        food_x, food_y = self.food
        state[food_y, food_x, 2] = 1  # food

        return state

    def is_collision(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size:
            return True
        if y < 0 or y >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False

    def step(self, action):
        if self.done:
            return self.get_state(), True

        head_x, head_y = self.snake[-1]

        if action == UP:
            head_y -= 1
        elif action == DOWN:
            head_y += 1
        elif action == LEFT:
            head_x -= 1
        elif action == RIGHT:
            head_x += 1

        new_head = (head_x, head_y)

        if self.is_collision(new_head):
            self.done = True
            return self.get_state(), True

        self.snake.append(new_head)

        if new_head == self.food:
            self.food = self.spawn_food()
        else:
            self.snake.pop(0)

        return self.get_state(), False


# -------------------
# Smarter Policy
# -------------------
def choose_smart_action(game):
    head_x, head_y = game.snake[-1]
    food_x, food_y = game.food

    best_action = None
    best_distance = float("inf")

    safe_actions = []

    for action in ACTIONS:
        x, y = head_x, head_y

        if action == UP:
            y -= 1
        elif action == DOWN:
            y += 1
        elif action == LEFT:
            x -= 1
        elif action == RIGHT:
            x += 1

        new_pos = (x, y)

        if not game.is_collision(new_pos):
            safe_actions.append(action)

            dist = abs(x - food_x) + abs(y - food_y)
            if dist < best_distance:
                best_distance = dist
                best_action = action

    if best_action is not None:
        return best_action

    if safe_actions:
        return np.random.choice(safe_actions)

    return np.random.choice(ACTIONS)


# -------------------
# Data Collection
# -------------------
def generate_dataset(dataset_size=DATASET_SIZE):

    game = SnakeGame(GRID_SIZE)

    states = []
    actions = []
    next_states = []
    dones = []

    while len(states) < dataset_size:

        current_state = game.get_state()
        action = choose_smart_action(game)
        next_state, done = game.step(action)

        states.append(current_state)
        actions.append(action)
        next_states.append(next_state)
        dones.append(done)

        if done:
            game.reset()

        if len(states) % 5000 == 0:
            print(f"Collected {len(states)} transitions")

    np.savez_compressed(
        OUTPUT_FILE,
        states=np.array(states),
        actions=np.array(actions),
        next_states=np.array(next_states),
        dones=np.array(dones)
    )

    print(f"\nDataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()