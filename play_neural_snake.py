import pygame
import torch
import numpy as np

from train_world_model import WorldModel

# Config
GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = WorldModel().to(device)
model.load_state_dict(torch.load("world_model.pt", map_location=device))
model.eval()

print("Model imported and loaded.")

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Learned Snake (World Model)")
clock = pygame.time.Clock()


def create_initial_state():
    state = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Snake head in center
    state[0, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0

    # Random food
    fx, fy = np.random.randint(0, GRID_SIZE, size=2)
    state[2, fx, fy] = 1.0

    return state


current_state = create_initial_state()


def draw_board(state):
    screen.fill((0, 0, 0))

    head = state[0]
    body = state[1]
    food = state[2]

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(
                j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE
            )

            if head[i, j] > 0.5:
                pygame.draw.rect(screen, (0, 255, 0), rect)
            elif body[i, j] > 0.5:
                pygame.draw.rect(screen, (0, 150, 0), rect)
            elif food[i, j] > 0.5:
                pygame.draw.rect(screen, (255, 0, 0), rect)

            pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    pygame.display.flip()


action = 0


def get_action_from_key(key):
    if key == pygame.K_UP:
        return 0
    if key == pygame.K_DOWN:
        return 1
    if key == pygame.K_LEFT:
        return 2
    if key == pygame.K_RIGHT:
        return 3
    return None


# Game loop
running = True

while running:
    clock.tick(6)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            new_action = get_action_from_key(event.key)
            if new_action is not None:
                action = new_action

    state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(device)
    action_tensor = torch.tensor([action], dtype=torch.long).to(device)

    with torch.no_grad():
        next_state = model(state_tensor, action_tensor)

    next_state = next_state.squeeze(0).cpu().numpy()

    # Force single head cell (stability fix)
    head_flat = next_state[0].reshape(-1)
    head_idx = np.argmax(head_flat)
    head_map = np.zeros_like(next_state[0])
    head_map.flat[head_idx] = 1.0
    next_state[0] = head_map

    # Threshold others
    next_state[1:] = (next_state[1:] > 0.5).astype(np.float32)

    current_state = next_state

    draw_board(current_state)

pygame.quit()
