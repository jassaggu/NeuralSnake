# eval_tools.py
from collections import deque

import numpy as np

from generate_truth_dataset import UP, DOWN, LEFT, RIGHT
import random

CHANNELS = 3
GRID_SIZE = 10

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]


# Reconstructs snake from unordered set of body cells and a head position, works on all snake shapes
def reconstruct_snake(body_positions, head):
    if len(body_positions) == 0:
        return [head]

    body_set = set(body_positions)
    all_cells = body_set | {head}

    def neighbours(pos):
        x, y = pos
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def snake_neighbours(pos):
        return [n for n in neighbours(pos) if n in all_cells]

    # Find the tail
    tail = None
    for cell in body_set:
        if len(snake_neighbours(cell)) == 1:
            tail = cell
            break

    # If no tail is found, pick an arbitrary body cell (in case of bad state)
    if tail is None:
        tail = next(iter(body_set))

    snake = [tail]
    visited = {tail}

    while True:
        current = snake[-1]
        unvisited = [n for n in snake_neighbours(current) if n not in visited]

        if not unvisited:
            break

        nxt = unvisited[0]
        snake.append(nxt)
        visited.add(nxt)

    # Check validity
    if len(snake) != len(body_positions) + 1:
        return None
    if snake[-1] != head:
        snake.reverse()
        if snake[-1] != head:
            return None

    return snake


# Get the next "ground truth" logical frame
def get_next_logical_frame(state, action):
    H, W, C = state.shape

    if C != CHANNELS:
        return None, True

    head_positions = np.argwhere(state[:, :, 1] == 1)
    body_positions = np.argwhere(state[:, :, 0] == 1)
    food_positions = np.argwhere(state[:, :, 2] == 1)

    if len(head_positions) != 1 or len(food_positions) != 1:
        return None, True

    head = tuple(head_positions[0][::-1])  # (x, y)
    body = [tuple(pos[::-1]) for pos in body_positions]
    food = tuple(food_positions[0][::-1])

    snake = reconstruct_snake(body, head)
    if snake is None:
        return None, True

    x, y = head

    if action == UP:
        y -= 1
    elif action == DOWN:
        y += 1
    elif action == LEFT:
        x -= 1
    elif action == RIGHT:
        x += 1
    else:
        return None, True

    new_head = (x, y)

    # Collision
    if x < 0 or x >= W or y < 0 or y >= H or new_head in snake:
        return state.copy(), True

    snake.append(new_head)

    food_eaten = (new_head == food)
    if not food_eaten:
        snake.pop(0)

    next_state = np.zeros_like(state)

    for bx, by in snake[:-1]:
        next_state[by, bx, 0] = 1

    hx, hy = snake[-1]
    next_state[hy, hx, 1] = 1

    if food_eaten:
        free_cells = [
            (x, y)
            for y in range(H)
            for x in range(W)
            if (x, y) not in snake
        ]

        if not free_cells:
            return state.copy(), True

        food = free_cells[np.random.randint(len(free_cells))]

    fx, fy = food
    next_state[fy, fx, 2] = 1

    return next_state, False


# Returns a procedurally generated Snake state of any length
def generate_random_snake_state(grid_size=GRID_SIZE, min_length=1, max_length=8):
    H = W = grid_size
    length = random.randint(min_length, max_length)

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dr, dc)

    for _ in range(1000):  # retry if generation fails
        offset = round((grid_size / 2) - 2)
        r = random.randint(offset, H - offset)
        c = random.randint(offset, W - offset)

        snake = [(r, c)]
        occupied = {(r, c)}
        success = True

        # Initialise direction randomly
        prev_dir = random.choice(DIRECTIONS)
        turn_cooldown = 0

        for _ in range(length - 1):
            hr, hc = snake[-1]

            candidates = []
            weights = []

            for dr, dc in DIRECTIONS:
                nr, nc = hr + dr, hc + dc

                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in occupied:
                    continue

                candidates.append((nr, nc))

                # Direction bias
                if (dr, dc) == prev_dir:
                    weight = 5.0  # Strongly prefer straight
                else:
                    if turn_cooldown > 0:
                        weight = 0.5  # Discourage immediate turns
                    else:
                        weight = 1.0  # Allow occasional turns

                weights.append(weight)

            if not candidates:
                success = False
                break

            # Weighted random choice
            idx = random.choices(range(len(candidates)), weights=weights)[0]
            nxt = candidates[idx]

            # Update direction
            new_dir = (nxt[0] - hr, nxt[1] - hc)

            if new_dir != prev_dir:
                turn_cooldown = 2  # For spacing between turns to avoid overcongestion
            else:
                turn_cooldown = max(0, turn_cooldown - 1)

            prev_dir = new_dir

            snake.append(nxt)
            occupied.add(nxt)

        if not success:
            continue

        # Place food in a free cell
        free_cells = [(r2, c2) for r2 in range(H) for c2 in range(W)
                      if (r2, c2) not in occupied]

        if not free_cells:
            continue

        food = random.choice(free_cells)

        # Build state array (channels: 0 = body, 1 = head, 2 = food)
        state = np.zeros((3, H, W), dtype=np.float32)

        for seg in snake[:-1]:
            state[0, seg[0], seg[1]] = 1.0

        head = snake[-1]
        state[1, head[0], head[1]] = 1.0
        state[2, food[0], food[1]] = 1.0

        return state, snake, food

    raise RuntimeError("Could not generate a valid random snake state after 1000 attempts.")


def flatten_logits(logits):
    # Helper to get shape compatibility between models
    # U-Net: (B, 1, H, W) -> (B, N)
    # Transformer/Baseline: (B, N) -> (B, N)
    if logits.dim() == 4:
        return logits.view(logits.size(0), -1)

    return logits


# Get the next "ground truth" logical frame
def rules_based_next_state(state_np, action, grid_size=GRID_SIZE):
    H = W = grid_size
    body_map = (state_np[0] > 0.5)
    head_map = (state_np[1] > 0.5)
    food_map = (state_np[2] > 0.5)

    head_pos = tuple(np.argwhere(head_map)[0])
    food_pos = tuple(np.argwhere(food_map)[0])

    hr, hc = head_pos
    if action == UP:
        hr -= 1
    elif action == DOWN:
        hr += 1
    elif action == LEFT:
        hc -= 1
    elif action == RIGHT:
        hc += 1
    new_head = (hr, hc)

    occupied = set(map(tuple, np.argwhere(head_map | body_map)))

    if not (0 <= hr < H and 0 <= hc < W) or new_head in occupied:
        return state_np.copy(), True

    ate_food = (new_head == food_pos)

    tail_pos = None
    if not ate_food and body_map.any():
        dist = {head_pos: 0}
        queue = deque([head_pos])
        while queue:
            pos = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (pos[0] + dr, pos[1] + dc)
                if nb not in dist and nb in occupied:
                    dist[nb] = dist[pos] + 1
                    queue.append(nb)
        tail_pos = max(dist, key=dist.get)

    next_state = np.zeros((3, H, W), dtype=np.float32)
    new_body = occupied.copy()
    if tail_pos is not None:
        new_body.discard(tail_pos)
    new_body.discard(new_head)

    for (r, c) in new_body:
        next_state[0, r, c] = 1.0
    next_state[1, new_head[0], new_head[1]] = 1.0

    if ate_food:
        all_occ = new_body | {new_head}
        free = [(r2, c2) for r2 in range(H) for c2 in range(W)
                if (r2, c2) not in all_occ]
        if free:
            fr, fc = random.choice(free)
            next_state[2, fr, fc] = 1.0
    else:
        next_state[2, food_pos[0], food_pos[1]] = 1.0

    return next_state, False
