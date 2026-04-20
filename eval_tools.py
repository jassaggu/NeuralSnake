# eval_tools.py
import numpy as np
from generate_truth_dataset import SnakeGame, GRID_SIZE, UP, DOWN, LEFT, RIGHT, ACTIONS
import random

CHANNELS = 3  # must match your state encoding


def reconstruct_snake(body_positions, head):
    """
    Reconstruct an ordered snake list from unordered body cells and a known head.
    Returns list ordered tail -> head, or None if reconstruction fails.

    Fix over the original greedy approach: instead of walking from the head and
    greedily picking any neighbour (which fails on coiled snakes where a cell
    has multiple body neighbours), we first locate the tail — the unique body
    cell with exactly one snake-neighbour — then walk the chain from tail to
    head. At every interior cell there is exactly one unvisited neighbour, so
    the walk is unambiguous regardless of snake shape.

    Edge case: length-1 snake (head only, no body cells) returns [head].
    """
    if len(body_positions) == 0:
        return [head]

    body_set = set(body_positions)
    all_cells = body_set | {head}

    def neighbours(pos):
        x, y = pos
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def snake_neighbours(pos):
        return [n for n in neighbours(pos) if n in all_cells]

    # Find the tail: the body cell with exactly one snake-neighbour.
    # (The head also has one neighbour when the snake is length 2, so we
    # restrict the search to body cells only.)
    tail = None
    for cell in body_set:
        if len(snake_neighbours(cell)) == 1:
            tail = cell
            break

    # Fallback: if no tail found (e.g. snake forms a loop due to bad state),
    # just pick an arbitrary body cell — reconstruction may still fail the
    # safety check below, which is the correct behaviour.
    if tail is None:
        tail = next(iter(body_set))

    # Walk from tail to head
    snake = [tail]
    visited = {tail}

    while True:
        current = snake[-1]
        unvisited = [n for n in snake_neighbours(current) if n not in visited]

        if not unvisited:
            break

        # There should be exactly one unvisited neighbour at every step
        # (two only at the tail, but we've already visited that direction).
        nxt = unvisited[0]
        snake.append(nxt)
        visited.add(nxt)

    # Safety check: reconstructed chain must contain all body cells + head
    if len(snake) != len(body_positions) + 1:
        return None

    # Ensure head is last (tail -> head order)
    if snake[-1] != head:
        snake.reverse()
        if snake[-1] != head:
            return None

    return snake  # tail -> head


def get_next_logical_frame(state, action):
    """
    Returns next logical frame.
    NOTE: Food channel is generated but should be IGNORED during comparison.
    """

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

    # collision
    if (
            x < 0 or x >= W or
            y < 0 or y >= H or
            new_head in snake
    ):
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


# -------------------------
# Unseen State  tion
# -------------------------
def generate_random_snake_state(grid_size=GRID_SIZE, min_length=1, max_length=8):
    """
    Procedurally generates a random, valid Snake state as a (3, H, W) numpy array.

    Modified to favour straighter snakes:
    - Bias towards continuing in the same direction
    - Prevents rapid consecutive turns (turn cooldown)

    Returns:
        state  : np.ndarray (3, grid_size, grid_size)  float32
        snake  : list of (row, col) tuples, tail-first, head last
        food   : (row, col) tuple
    """
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

                # ---- Direction bias ----
                if (dr, dc) == prev_dir:
                    weight = 5.0  # strongly prefer straight
                else:
                    if turn_cooldown > 0:
                        weight = 0.5  # discourage immediate turns
                    else:
                        weight = 1.0  # allow occasional turns

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
                turn_cooldown = 2  # enforce spacing between turns
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

        # Build state array (channels: 0=body, 1=head, 2=food)
        state = np.zeros((3, H, W), dtype=np.float32)

        for seg in snake[:-1]:
            state[0, seg[0], seg[1]] = 1.0

        head = snake[-1]
        state[1, head[0], head[1]] = 1.0
        state[2, food[0], food[1]] = 1.0

        return state, snake, food

    raise RuntimeError("Could not generate a valid random snake state after 1000 attempts.")