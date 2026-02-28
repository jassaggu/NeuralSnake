import pygame
import torch
import numpy as np

from train_transformer_model import TransformerWorldModel

# Config
GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- Load Correct Model ----
model = TransformerWorldModel()
model.load_state_dict(torch.load("transformer_world_model.pt", map_location=device))
model.to(device)
model.eval()

print("Transformer model loaded.")

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Learned Snake (Transformer World Model)")
clock = pygame.time.Clock()


# ---- Correct Channel Ordering ----
# channel 0 = body
# channel 1 = head
# channel 2 = food

def create_initial_state():
    state = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2

    state[0, cx, cy] = 1.0  # body
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

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if body[i, j] > 0.5:
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
    # HEAD
    head_idx = torch.argmax(head_logits, dim=1)
    head_map = torch.zeros((1, NUM_CELLS), device=device)
    head_map[0, head_idx] = 1.0

    # BODY
    body_map = (torch.sigmoid(body_logits) > 0.5).float()

    # FOOD
    food_idx = torch.argmax(food_logits, dim=1)
    food_map = torch.zeros((1, NUM_CELLS), device=device)
    food_map[0, food_idx] = 1.0

    next_state = torch.stack([
        body_map.view(1, GRID_SIZE, GRID_SIZE),
        head_map.view(1, GRID_SIZE, GRID_SIZE),
        food_map.view(1, GRID_SIZE, GRID_SIZE)
    ], dim=1)

    return next_state


current_state = create_initial_state()
action = 0
running = True

while running:
    clock.tick(6)

    # ---- Event Handling ----
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

    # ---- Use done_logit for termination ----
    done_prob = torch.sigmoid(done_logit)
    done_pred = (done_prob > 0.5).item()

    if done_pred:
        print("Model predicted termination. Resetting environment.")
        current_state = create_initial_state()
        action = 0
        draw_board(current_state)
        continue

    # ---- Otherwise continue rollout ----
    next_state = reconstruct_state(head_logits, body_logits, food_logits)
    current_state = next_state.squeeze(0).cpu().numpy()

    draw_board(current_state)

pygame.quit()