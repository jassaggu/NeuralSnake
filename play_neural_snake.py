import pygame
import torch
import numpy as np

from train_world_model import WorldModel
from train_UNet_model import ResidualUNetWorldModel

# Config
GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
# model = WorldModel()
model = ResidualUNetWorldModel()

model.to(device)
model.load_state_dict(torch.load("residual_unet_world_model.pt", map_location=device))
model.eval()

print("Model imported and loaded.")

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Learned Snake (World Model)")
clock = pygame.time.Clock()


def create_initial_state():
    state = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    state[0, cx, cy] = 1.0  # head

    fx, fy = np.random.randint(0, GRID_SIZE, size=2)
    state[2, fx, fy] = 1.0  # food

    return state


def draw_board(state):
    screen.fill((0, 0, 0))

    head = state[0]
    body = state[1]
    food = state[2]

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if head[i, j] > 0.5:
                pygame.draw.rect(screen, (0, 255, 0), rect)
            elif body[i, j] > 0.5:
                pygame.draw.rect(screen, (0, 150, 0), rect)
            elif food[i, j] > 0.5:
                pygame.draw.rect(screen, (255, 0, 0), rect)

            pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    pygame.display.flip()


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


current_state = create_initial_state()
action = 0  # default direction
running = True

while running:
    clock.tick(6)

    # --- Handle Input ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            new_action = get_action_from_key(event.key)
            if new_action is not None:
                action = new_action

    # --- Model Step ---
    state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(device)
    action_tensor = torch.tensor([action], dtype=torch.long).to(device)

    with torch.no_grad():
        head_logits, body_logits, food_logits = model(state_tensor, action_tensor)

    # Remove batch dim
    head_logits = head_logits.squeeze(0)  # (1,H,W)
    body_logits = body_logits.squeeze(0)
    food_logits = food_logits.squeeze(0)

    # ---- HEAD (softmax over grid) ----
    H, W = head_logits.shape[1:]
    head_flat = head_logits.view(-1)
    head_probs = torch.softmax(head_flat, dim=0)
    head_idx = torch.argmax(head_probs)

    head_map = torch.zeros_like(head_flat)
    head_map[head_idx] = 1.0
    head_map = head_map.view(1, H, W)

    # ---- BODY & FOOD (sigmoid + threshold) ----
    body_map = torch.sigmoid(body_logits)
    food_map = torch.sigmoid(food_logits)

    body_map = (body_map > 0.5).float()
    food_map = (food_map > 0.5).float()

    # ---- Stack back into state ----
    next_state = torch.cat([body_map, head_map, food_map], dim=0)

    next_state = next_state.cpu().numpy()

    # --- Post Processing ---

    # Force single head cell
    head_flat = next_state[0].reshape(-1)
    head_idx = np.argmax(head_flat)
    head_map = np.zeros_like(next_state[0])
    head_map.flat[head_idx] = 1.0
    next_state[0] = head_map

    # Threshold body and food separately
    next_state[1] = (next_state[1] > 0.5).astype(np.float32)
    next_state[2] = (next_state[2] > 0.5).astype(np.float32)

    current_state = next_state

    # --- Render ---
    draw_board(current_state)

pygame.quit()