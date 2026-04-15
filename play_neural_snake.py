import pygame
import torch
import numpy as np

from select_model import load_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Config
MODEL = "unet"      # baseline, unet, transformer
GRID_SIZE = 100      # change this freely now
CELL_SIZE = 10
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

model = load_model(MODEL, device)

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Learned Snake (Neural World Model)")
clock = pygame.time.Clock()


# channel 0 = body
# channel 1 = head
# channel 2 = food

def create_initial_state():
    state = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2

    state[1, cx, cy] = 1.0  # head

    while True:
        fx, fy = np.random.randint(0, GRID_SIZE, size=2)
        if (fx, fy) != (cx, cy):
            break

    state[2, fx, fy] = 1.0  # food

    return state


def draw_board(state):
    screen.fill((0, 0, 0))

    body = state[0]
    head = state[1]
    food = state[2]

    H, W = body.shape

    for i in range(H):
        for j in range(W):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if body[i, j] > 0.1:
                pygame.draw.rect(screen, (0, 150, 0), rect)

            if head[i, j] > 0.5:
                pygame.draw.rect(screen, (0, 255, 0), rect)

            if food[i, j] > 0.5:
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


def reconstruct_state(head_logits, body_logits, food_logits):
    # ---- Handle both flattened and spatial outputs ----
    if head_logits.dim() == 4:
        # (B, 1, H, W) → (B, N)
        B, _, H, W = head_logits.shape
        N = H * W

        head_logits = head_logits.view(B, -1)
        body_logits = body_logits.view(B, -1)
        food_logits = food_logits.view(B, -1)

    elif head_logits.dim() == 2:
        # (B, N)
        B, N = head_logits.shape
        H = W = int(np.sqrt(N))
    else:
        raise ValueError(f"Unexpected logits shape: {head_logits.shape}")

    # ---- HEAD ----
    head_idx = torch.argmax(head_logits, dim=1)
    head_map = torch.zeros((B, N), device=device)
    head_map[torch.arange(B), head_idx] = 1.0

    # ---- BODY ----
    body_map = (torch.sigmoid(body_logits) > 0.5).float()

    # ---- FOOD ----
    food_idx = torch.argmax(food_logits, dim=1)
    food_map = torch.zeros((B, N), device=device)
    food_map[torch.arange(B), food_idx] = 1.0

    # ---- Reconstruct grid ----
    next_state = torch.stack([
        body_map.view(B, H, W),
        head_map.view(B, H, W),
        food_map.view(B, H, W)
    ], dim=1)

    return next_state


current_state = create_initial_state()
action = 0
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

    state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(device)
    action_tensor = torch.tensor([action], dtype=torch.long).to(device)

    with torch.no_grad():
        head_logits, body_logits, food_logits, done_logit = model(state_tensor, action_tensor)

    # termination
    done_prob = torch.sigmoid(done_logit)
    done_pred = (done_prob > 0.5).item()

    if done_pred:
        print("Model predicted termination. Resetting environment.")
        current_state = create_initial_state()
        action = 0
        draw_board(current_state)
        continue

    next_state = reconstruct_state(head_logits, body_logits, food_logits)
    current_state = next_state.squeeze(0).cpu().numpy()

    draw_board(current_state)

pygame.quit()