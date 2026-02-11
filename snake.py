import pygame
import numpy as np

# --- Config ---
GRID_SIZE = 10  # 10x10 grid
CELL_SIZE = 20  # pixel size for display
CHANNELS = 3  # body, head, food
DATASET_SIZE = 5000  # number of transitions to collect
OUTPUT_FILE = "snake_transitions.npz"

# action encoding
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

HEIGHT, WIDTH = 100, 100
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Collecting data...")

clock = pygame.time.Clock()


# --- Game logic/ground truth ---
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
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in self.snake:
                return pos

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size, CHANNELS), dtype=np.float32)
        for x, y in self.snake[:-1]:
            state[y, x, 0] = 1  # body
        head_x, head_y = self.snake[-1]
        state[head_y, head_x, 1] = 1    # head
        food_x, food_y = self.food
        state[food_y, food_x, 2] = 1    # food
        return state

    def step(self, action):
        if self.done:
            return self.get_state(), True

        # Determine new head position
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

        # Check collisions
        if (head_x < 0 or head_x >= self.grid_size or
                head_y < 0 or head_y >= self.grid_size or
                new_head in self.snake):
            self.done = True
            return self.get_state(), True

        # Move snake
        self.snake.append(new_head)
        if new_head == self.food:
            self.food = self.spawn_food()
        else:
            self.snake.pop(0)

        return self.get_state(), False


def draw_state(screen, state, cell_size):
    screen.fill((0, 0, 0))  # clear screen

    h, w, _ = state.shape

    for y in range(h):
        for x in range(w):
            body, head, food = state[y, x]

            if body:
                colour = (0, 200, 0)    # green body
            elif head:
                colour = (0, 255, 0)    # bright green head
            elif food:
                colour = (200, 0, 0)    # red food
            else:
                continue

            pygame.draw.rect(
                screen, colour,(x * cell_size, y * cell_size, cell_size, cell_size)
            )

    pygame.display.flip()


# ------------------------------
# DATA COLLECTION
# ------------------------------
def generate_dataset(dataset_size=DATASET_SIZE, render=True):
    pygame.init()
    screen = pygame.display.set_mode(
        (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
    )
    clock = pygame.time.Clock()

    game = SnakeGame(GRID_SIZE)

    states, actions, next_states, dones = [], [], [], []

    while len(states) < dataset_size:

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current_state = game.get_state()

        if render:
            draw_state(screen, current_state, CELL_SIZE)
            clock.tick(5)   # cap render speed

        action = np.random.choice(ACTIONS)
        next_state, done = game.step(action)

        states.append(current_state)
        actions.append(action)
        next_states.append(next_state)
        dones.append(done)

        if done:
            game.reset()

    pygame.quit()

    np.savez_compressed(
        OUTPUT_FILE,
        states=np.array(states),
        actions=np.array(actions),
        next_states=np.array(next_states),
        dones=np.array(dones)
    )

    print(f"Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
