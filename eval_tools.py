# eval_tools.py
import numpy as np
from generate_truth_dataset import SnakeGame, GRID_SIZE, UP, DOWN, LEFT, RIGHT, ACTIONS

CHANNELS = 3  # must match your state encoding


def get_next_logical_frame(state, action):
    """
    Given a state/frame and an action, returns the next logical frame and done flag.

    Parameters:
        state: np.ndarray of shape (GRID_SIZE, GRID_SIZE, 3)
        action: int, one of [UP, DOWN, LEFT, RIGHT]

    Returns:
        next_state: np.ndarray of shape (GRID_SIZE, GRID_SIZE, 3) or None if invalid
        done: bool (-1 if invalid)
    """
    # Validate shape
    if state.shape != (GRID_SIZE, GRID_SIZE, CHANNELS):
        return None, -1

    # Extract head, body, food
    head_positions = np.argwhere(state[:, :, 1] == 1)
    body_positions = np.argwhere(state[:, :, 0] == 1)
    food_positions = np.argwhere(state[:, :, 2] == 1)

    # Check validity
    if len(head_positions) != 1 or len(food_positions) != 1:
        return None, -1

    head = tuple(head_positions[0][::-1])  # (x, y)
    body = [tuple(pos[::-1]) for pos in body_positions]
    food = tuple(food_positions[0][::-1])

    # Reconstruct snake
    snake = body + [head]

    # Compute new head position
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
        return None, -1

    new_head = (x, y)

    # Check collisions
    if (
            x < 0 or x >= GRID_SIZE
            or y < 0 or y >= GRID_SIZE
            or new_head in snake
    ):
        done = True
        return state.copy(), done

    # Move snake
    snake.append(new_head)
    if new_head == food:
        # Food eaten -> spawn new food at random free cell
        free_cells = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if (i, j) not in snake
        ]
        if not free_cells:
            # snake filled entire grid, game done
            done = True
            return state.copy(), done
        food = free_cells[np.random.randint(len(free_cells))]
    else:
        # Remove tail
        snake.pop(0)

    # Construct next state
    next_state = np.zeros_like(state)
    for bx, by in snake[:-1]:
        next_state[by, bx, 0] = 1  # body
    hx, hy = snake[-1]
    next_state[hy, hx, 1] = 1  # head
    fx, fy = food
    next_state[fy, fx, 2] = 1  # food

    done = False
    return next_state, done