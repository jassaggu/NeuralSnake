# eval_tools.py
import numpy as np
from generate_truth_dataset import SnakeGame, GRID_SIZE, UP, DOWN, LEFT, RIGHT, ACTIONS

CHANNELS = 3  # must match your state encoding


def reconstruct_snake(body_positions, head):
    """
    Reconstruct ordered snake from unordered body cells.
    Returns list from tail -> head.
    """
    body_set = set(body_positions)
    snake = [head]

    current = head

    while True:
        x, y = current

        neighbours = [
            (x+1, y),
            (x-1, y),
            (x, y+1),
            (x, y-1)
        ]

        next_cell = None
        for n in neighbours:
            if n in body_set:
                next_cell = n
                break

        if next_cell is None:
            break

        snake.append(next_cell)
        body_set.remove(next_cell)
        current = next_cell

    snake.reverse()  # tail -> head

    # Optional safety check
    if len(snake) != len(body_positions) + 1:
        return None

    return snake


def get_next_logical_frame(state, action):
    """
    Returns next logical frame.
    NOTE: Food channel is generated but should be IGNORED during comparison.
    """

    # Validate shape
    if state.shape != (GRID_SIZE, GRID_SIZE, CHANNELS):
        return None, -1

    # Extract channels
    head_positions = np.argwhere(state[:, :, 1] == 1)
    body_positions = np.argwhere(state[:, :, 0] == 1)
    food_positions = np.argwhere(state[:, :, 2] == 1)

    # Basic validity
    if len(head_positions) != 1 or len(food_positions) != 1:
        return None, -1

    head = tuple(head_positions[0][::-1])  # (x, y)
    body = [tuple(pos[::-1]) for pos in body_positions]
    food = tuple(food_positions[0][::-1])

    snake = reconstruct_snake(body, head)
    if snake is None:
        return None, -1

    # Compute new head
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

    # Collision check
    if (
        x < 0 or x >= GRID_SIZE or
        y < 0 or y >= GRID_SIZE or
        new_head in snake
    ):
        return state.copy(), True

    # Move snake
    snake.append(new_head)

    food_eaten = (new_head == food)

    if not food_eaten:
        snake.pop(0)  # remove tail

    # -------- Construct next state --------
    next_state = np.zeros_like(state)

    for bx, by in snake[:-1]:
        next_state[by, bx, 0] = 1  # body

    hx, hy = snake[-1]
    next_state[hy, hx, 1] = 1  # head

    # Food is re-generated but SHOULD NOT be used for divergence
    if food_eaten:
        free_cells = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if (i, j) not in snake
        ]

        if not free_cells:
            return state.copy(), True

        food = free_cells[np.random.randint(len(free_cells))]

    fx, fy = food
    next_state[fy, fx, 2] = 1

    return next_state, False