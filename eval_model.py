import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import deque
import random

from select_model import load_model
from eval_tools import get_next_logical_frame

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "snake_transitions.npz"
BATCH_SIZE = 128
GRID_SIZE = 10

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]


# -------------------------
# Dataset
# -------------------------
class SnakeDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.states = data["states"]
        self.actions = data["actions"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx]).permute(2, 0, 1).float()
        next_state = torch.tensor(self.next_states[idx]).permute(2, 0, 1).float()
        action = torch.tensor(self.actions[idx]).long()
        done = torch.tensor(self.dones[idx]).float()
        return state, action, next_state, done


# -------------------------
# Helper
# -------------------------
def flatten_logits(logits):
    """
    Ensures compatibility across:
    - U-Net:               (B, 1, H, W)
    - Transformer/Baseline: (B, N)
    """
    if logits.dim() == 4:
        return logits.view(logits.size(0), -1)
    return logits


# -------------------------
# State Validity Check
# -------------------------
def is_valid_state(state_tensor):
    """
    Checks whether a (3, H, W) state tensor represents a legal Snake game state.

    Channel layout (matching generate_data.py):
        0 = body
        1 = head
        2 = food

    Rules:
        1. Exactly one head cell.
        2. Exactly one food cell.
        3. No overlap between any pair of channels.
        4. All body cells (if any) are connected to the head via 4-connectivity,
           forming a single component — no disconnected body islands.
    """
    if isinstance(state_tensor, torch.Tensor):
        state = state_tensor.cpu().numpy()
    else:
        state = state_tensor  # already numpy

    body = (state[0] > 0.5).astype(np.uint8)  # channel 0
    head = (state[1] > 0.5).astype(np.uint8)  # channel 1
    food = (state[2] > 0.5).astype(np.uint8)  # channel 2

    # Rule 1 — exactly one head
    if head.sum() != 1:
        return False

    # Rule 2 — exactly one food
    if food.sum() != 1:
        return False

    # Rule 3 — no channel overlap
    if np.any((head + body) > 1):
        return False
    if np.any((head + food) > 1):
        return False
    if np.any((body + food) > 1):
        return False

    # Rule 4 — body cells (if any) must all be 4-connected to the head
    body_count = int(body.sum())
    if body_count > 0:
        H, W = head.shape
        head_pos = tuple(np.argwhere(head == 1)[0])

        # BFS from head over (head | body) cells
        occupied = (head | body).astype(bool)
        visited = set()
        queue = deque([head_pos])
        visited.add(head_pos)

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if occupied[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # Every body cell must have been reached from the head
        body_positions = set(map(tuple, np.argwhere(body == 1)))
        if not body_positions.issubset(visited):
            return False

    return True


# -------------------------
# Unseen State Generation
# -------------------------
def generate_random_snake_state(grid_size=GRID_SIZE, min_length=1, max_length=8):
    """
    Procedurally generates a random, valid Snake state as a (3, H, W) numpy array.

    Snake is grown by random walk from a random starting position.
    Food is placed on a free cell.

    Returns:
        state  : np.ndarray (3, grid_size, grid_size)  float32
        snake  : list of (row, col) tuples, tail-first, head last
        food   : (row, col) tuple
    """
    H = W = grid_size
    length = random.randint(min_length, max_length)

    for _ in range(1000):  # retry if random walk gets stuck
        r = random.randint(0, H - 1)
        c = random.randint(0, W - 1)
        snake = [(r, c)]
        occupied = {(r, c)}
        success = True

        for _ in range(length - 1):
            neighbours = []
            hr, hc = snake[-1]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = hr + dr, hc + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in occupied:
                    neighbours.append((nr, nc))
            if not neighbours:
                success = False
                break
            nxt = random.choice(neighbours)
            snake.append(nxt)
            occupied.add(nxt)

        if not success:
            continue

        free_cells = [(r2, c2) for r2 in range(H) for c2 in range(W)
                      if (r2, c2) not in occupied]
        if not free_cells:
            continue

        food = random.choice(free_cells)

        # Build state array  (channels: 0=body, 1=head, 2=food)
        state = np.zeros((3, H, W), dtype=np.float32)
        for seg in snake[:-1]:  # body (all but head)
            state[0, seg[0], seg[1]] = 1.0
        head = snake[-1]
        state[1, head[0], head[1]] = 1.0
        state[2, food[0], food[1]] = 1.0

        return state, snake, food

    raise RuntimeError("Could not generate a valid random snake state after 1000 attempts.")


def rules_based_next_state(state_np, action, grid_size=GRID_SIZE):
    """
    Given a (3, H, W) state array and an integer action, returns the ground-truth
    next state and done flag by applying the true Snake game rules.

    The snake list is reconstructed from the spatial grid via BFS from the head,
    which gives us the set of occupied cells; we approximate tail removal by
    picking the body cell furthest (graph distance) from the head as the tail.

    Args:
        state_np : np.ndarray (3, H, W)
        action   : int  (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        grid_size: int

    Returns:
        next_state : np.ndarray (3, H, W) float32
        done       : bool
    """
    H = W = grid_size
    body_map = (state_np[0] > 0.5)
    head_map = (state_np[1] > 0.5)
    food_map = (state_np[2] > 0.5)

    head_pos = tuple(np.argwhere(head_map)[0])  # (row, col)
    food_pos = tuple(np.argwhere(food_map)[0])

    # Compute new head position
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

    # Occupied cells = head + body
    occupied = set(map(tuple, np.argwhere(head_map | body_map)))

    # Collision check
    if not (0 <= hr < H and 0 <= hc < W) or new_head in occupied:
        # On death return current state and done=True
        return state_np.copy(), True

    ate_food = (new_head == food_pos)

    # Find the tail — the body cell with the greatest BFS distance from the head.
    # This correctly identifies which end to remove when not eating.
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

    # Build next state
    next_state = np.zeros((3, H, W), dtype=np.float32)

    # Copy existing body+head into body channel, then remove tail
    new_body = occupied.copy()
    if tail_pos is not None:
        new_body.discard(tail_pos)

    # New head is not a body cell
    new_body.discard(new_head)

    for (r, c) in new_body:
        next_state[0, r, c] = 1.0  # body

    next_state[1, new_head[0], new_head[1]] = 1.0  # head

    if ate_food:
        # Spawn new food on a free cell
        all_occupied = new_body | {new_head}
        free = [(r2, c2) for r2 in range(H) for c2 in range(W)
                if (r2, c2) not in all_occupied]
        if free:
            fr, fc = random.choice(free)
            next_state[2, fr, fc] = 1.0
        # (if board is full, no food — edge case)
    else:
        next_state[2, food_pos[0], food_pos[1]] = 1.0

    return next_state, False


# -------------------------
# Shared Metric Computation
# -------------------------
def compute_metrics(
        all_head_correct,
        all_done_correct,
        all_body_tp, all_body_fp, all_body_fn,
        all_fc_tp, all_fc_fp, all_fc_fn,
        all_food_static_correct, all_food_static_total,
        all_food_respawn_correct, all_food_respawn_total,
        all_illegal,
        total,
):
    head_acc = sum(all_head_correct) / total
    done_acc = sum(all_done_correct) / total

    tp = sum(all_body_tp)
    fp = sum(all_body_fp)
    fn = sum(all_body_fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    body_f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

    fc_tp = sum(all_fc_tp)
    fc_fp = sum(all_fc_fp)
    fc_fn = sum(all_fc_fn)
    fc_precision = fc_tp / (fc_tp + fc_fp) if (fc_tp + fc_fp) > 0 else 0.0
    fc_recall = fc_tp / (fc_tp + fc_fn) if (fc_tp + fc_fn) > 0 else 0.0
    fc_f1 = (2 * fc_precision * fc_recall / (fc_precision + fc_recall)
             if (fc_precision + fc_recall) > 0 else 0.0)
    fc_support = fc_tp + fc_fn

    fs_total = sum(all_food_static_total)
    food_static_acc = (sum(all_food_static_correct) / fs_total
                       if fs_total > 0 else float("nan"))

    fr_total = sum(all_food_respawn_total)
    food_respawn_acc = (sum(all_food_respawn_correct) / fr_total
                        if fr_total > 0 else float("nan"))

    illegal_rate = sum(all_illegal) / total

    return {
        "head_acc": head_acc,
        "done_acc": done_acc,
        "body_f1": body_f1,
        "fc_f1": fc_f1,
        "fc_support": fc_support,
        "food_static_acc": food_static_acc,
        "food_static_n": fs_total,
        "food_respawn_acc": food_respawn_acc,
        "food_respawn_n": fr_total,
        "illegal_rate": illegal_rate,
    }


def print_metrics(metrics, label):
    print(f"\n=== {label} ===")
    print(f"  Head Accuracy:              {metrics['head_acc']:.4f}")
    print(f"  Done Accuracy:              {metrics['done_acc']:.4f}")
    print(f"  Body F1 Score:              {metrics['body_f1']:.4f}")
    print(f"  Food Consumption F1:        {metrics['fc_f1']:.4f}  (support: {metrics['fc_support']})")
    print(f"  Food Static Accuracy:       {metrics['food_static_acc']:.4f}  (n={metrics['food_static_n']})")
    print(f"  Food Respawn Accuracy:      {metrics['food_respawn_acc']:.4f}  (n={metrics['food_respawn_n']})")
    print(f"  Illegal State Rate:         {metrics['illegal_rate']:.4f}")


def _accumulate_batch(
        states, actions, next_states, dones, model,
        all_head_correct, all_done_correct,
        all_body_tp, all_body_fp, all_body_fn,
        all_fc_tp, all_fc_fp, all_fc_fn,
        all_food_static_correct, all_food_static_total,
        all_food_respawn_correct, all_food_respawn_total,
        all_illegal,
):
    """
    Runs one batch through the model, decodes predictions, and accumulates
    metric accumulators in-place.
    """
    with torch.no_grad():
        head_logits, body_logits, food_logits, done_logit = model(states, actions)

    B = states.size(0)
    H = W = GRID_SIZE
    N = H * W

    head_flat = flatten_logits(head_logits)  # (B, N)
    food_flat = flatten_logits(food_logits)  # (B, N)
    body_flat = flatten_logits(body_logits)  # (B, N)

    pred_head_idx = head_flat.argmax(dim=1)
    pred_food_idx = food_flat.argmax(dim=1)
    pred_body_bin = (torch.sigmoid(body_flat) > 0.5).float()
    pred_done = (torch.sigmoid(done_logit) > 0.5).float()

    target_head = next_states[:, 1].view(B, -1)  # channel 1 = head
    target_food = next_states[:, 2].view(B, -1)  # channel 2 = food
    target_body = next_states[:, 0].view(B, -1)  # channel 0 = body

    gt_head_idx = target_head.argmax(dim=1)
    gt_food_idx = target_food.argmax(dim=1)

    # Head accuracy
    all_head_correct.extend((pred_head_idx == gt_head_idx).cpu().tolist())

    # Done accuracy
    all_done_correct.extend((pred_done.squeeze(-1) == dones).cpu().tolist())

    # Body F1 components
    tp = (pred_body_bin * target_body).sum(dim=1)
    fp = (pred_body_bin * (1 - target_body)).sum(dim=1)
    fn = ((1 - pred_body_bin) * target_body).sum(dim=1)
    all_body_tp.extend(tp.cpu().tolist())
    all_body_fp.extend(fp.cpu().tolist())
    all_body_fn.extend(fn.cpu().tolist())

    # Food consumption F1
    # Positive class = head lands on the current food cell.
    prev_food_idx = states[:, 2].view(B, -1).argmax(dim=1)  # food in current state
    gt_consumed = (gt_head_idx == prev_food_idx)
    pred_consumed = (pred_head_idx == prev_food_idx)

    fc_tp = (gt_consumed & pred_consumed).long()
    fc_fp = (~gt_consumed & pred_consumed).long()
    fc_fn = (gt_consumed & ~pred_consumed).long()
    all_fc_tp.extend(fc_tp.cpu().tolist())
    all_fc_fp.extend(fc_fp.cpu().tolist())
    all_fc_fn.extend(fc_fn.cpu().tolist())

    # Food Static Accuracy — non-eating steps only.
    # Food should be unchanged: gt_food_idx == prev_food_idx.
    # We check whether the model also kept it unchanged.
    static_mask = ~gt_consumed  # (B,) bool — transitions where no food was eaten
    if static_mask.any():
        pred_food_static_correct = (pred_food_idx[static_mask] == gt_food_idx[static_mask])
        all_food_static_correct.extend(pred_food_static_correct.cpu().tolist())
        all_food_static_total.extend([1] * int(static_mask.sum().item()))

    # Food Respawn Accuracy — eating steps only.
    # Food has moved to a new (stochastic) location; check if the model predicted it.
    respawn_mask = gt_consumed  # (B,) bool — transitions where food was eaten
    if respawn_mask.any():
        pred_food_respawn_correct = (pred_food_idx[respawn_mask] == gt_food_idx[respawn_mask])
        all_food_respawn_correct.extend(pred_food_respawn_correct.cpu().tolist())
        all_food_respawn_total.extend([1] * int(respawn_mask.sum().item()))

    # Illegal state rate — reconstruct predicted next state and validate
    pred_head_idx_np = pred_head_idx.cpu().numpy()
    pred_food_idx_np = pred_food_idx.cpu().numpy()
    pred_body_np = pred_body_bin.cpu().numpy()

    for i in range(B):
        pred_state = np.zeros((3, H, W), dtype=np.float32)
        ph = pred_head_idx_np[i]
        pred_state[1, ph // W, ph % W] = 1.0
        pf = pred_food_idx_np[i]
        pred_state[2, pf // W, pf % W] = 1.0
        pred_state[0] = pred_body_np[i].reshape(H, W)
        all_illegal.append(0 if is_valid_state(pred_state) else 1)


# -------------------------
# Evaluation 1 — Dataset
# -------------------------
def evaluate_with_dataset(model, loader):
    model.eval()

    acc = {k: [] for k in [
        "head", "done", "body_tp", "body_fp", "body_fn",
        "fc_tp", "fc_fp", "fc_fn",
        "food_static_correct", "food_static_total",
        "food_respawn_correct", "food_respawn_total",
        "illegal"
    ]}

    for states, actions, next_states, dones in loader:
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)

        _accumulate_batch(
            states, actions, next_states, dones, model,
            acc["head"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
            acc["food_static_correct"], acc["food_static_total"],
            acc["food_respawn_correct"], acc["food_respawn_total"],
            acc["illegal"],
        )

    total = len(acc["head"])
    metrics = compute_metrics(
        acc["head"], acc["done"],
        acc["body_tp"], acc["body_fp"], acc["body_fn"],
        acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
        acc["food_static_correct"], acc["food_static_total"],
        acc["food_respawn_correct"], acc["food_respawn_total"],
        acc["illegal"], total,
    )
    print_metrics(metrics, "Dataset Evaluation (held-out test split)")
    return metrics


# -------------------------
# Evaluation 2 — Unseen
# -------------------------
def evaluate_unseen(model, n_samples=2000, grid_size=GRID_SIZE):
    """
    Generates n_samples random valid Snake states, computes the ground-truth
    next state with rules_based_next_state, runs the model, and scores it.
    No data from the training dataset is used.
    """
    model.eval()

    acc = {k: [] for k in [
        "head", "done", "body_tp", "body_fp", "body_fn",
        "fc_tp", "fc_fp", "fc_fn",
        "food_static_correct", "food_static_total",
        "food_respawn_correct", "food_respawn_total",
        "illegal"
    ]}

    skipped = 0

    for _ in range(n_samples):
        try:
            state_np, _, _ = generate_random_snake_state(grid_size)
        except RuntimeError:
            skipped += 1
            continue

        action = random.choice(ACTIONS)
        next_state_np, done = rules_based_next_state(state_np, action, grid_size)

        # Convert to tensors with batch dim
        state_t = torch.tensor(state_np).unsqueeze(0).to(DEVICE)
        next_state_t = torch.tensor(next_state_np).unsqueeze(0).to(DEVICE)
        action_t = torch.tensor([action], dtype=torch.long).to(DEVICE)
        done_t = torch.tensor([float(done)]).to(DEVICE)

        _accumulate_batch(
            state_t, action_t, next_state_t, done_t, model,
            acc["head"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
            acc["food_static_correct"], acc["food_static_total"],
            acc["food_respawn_correct"], acc["food_respawn_total"],
            acc["illegal"],
        )

    if skipped:
        print(f"  (skipped {skipped} samples due to generation failures)")

    total = len(acc["head"])
    if total == 0:
        print("  No samples evaluated.")
        return {}

    metrics = compute_metrics(
        acc["head"], acc["done"],
        acc["body_tp"], acc["body_fp"], acc["body_fn"],
        acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
        acc["food_static_correct"], acc["food_static_total"],
        acc["food_respawn_correct"], acc["food_respawn_total"],
        acc["illegal"], total,
    )
    print_metrics(metrics, "Unseen State Evaluation (procedurally generated)")
    return metrics


# -------------------------
# Rollout Evaluation
# -------------------------
ROLLOUT_DIVERGENCE_STEPS = [1, 5, 10, 20, 30, 50]
ROLLOUT_MAX_STEPS = 50
ROLLOUT_N = 100
ROLLOUT_MIN_SNAKE_LENGTH = 4  # skip trivially short starting states
ROLLOUT_P_STRAIGHT = 0.6  # probability of continuing in current direction


def _biased_action(current_action, rng):
    """
    Sample the next action with a straight bias.

    With probability P_STRAIGHT continue in the current direction.
    The reverse direction is always excluded (illegal in Snake).
    Remaining probability is split equally between the two turn directions.

    Action opposites: UP<->DOWN, LEFT<->RIGHT
    """
    opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
    turns = [a for a in ACTIONS if a != opposite[current_action]]

    if rng.random() < ROLLOUT_P_STRAIGHT:
        return current_action

    # Choose uniformly from the two turn options (exclude straight and reverse)
    turn_options = [a for a in turns if a != current_action]
    return rng.choice(turn_options)


def _channel_iou(pred_channel, gt_channel):
    """
    Compute IoU between two binary (H, W) arrays.
    Returns 1.0 if both are all-zero (nothing to compare).
    """
    pred = pred_channel > 0.5
    gt = gt_channel > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _state_iou(pred_chw, gt_hwc, ate_food):
    """
    Mean IoU across channels, excluding food channel on eating steps
    (since food respawn is stochastic and cannot be predicted deterministically).

    pred_chw : np.ndarray (C, H, W)  — model prediction
    gt_hwc   : np.ndarray (H, W, C)  — ground truth from simulator
    ate_food : bool
    """
    channels = [0, 1] if ate_food else [0, 1, 2]
    ious = []
    for c in channels:
        pred_c = pred_chw[c]  # (H, W)
        gt_c = gt_hwc[:, :, c]  # (H, W)
        ious.append(_channel_iou(pred_c, gt_c))
    return float(np.mean(ious))


def _decode_model_output(head_logits, body_logits, food_logits):
    """
    Convert raw model outputs to a (3, H, W) binary numpy prediction.
    Compatible with U-Net (4D) and Transformer/Baseline (2D) outputs.
    """
    H = W = GRID_SIZE

    head_flat = flatten_logits(head_logits)  # (1, N)
    food_flat = flatten_logits(food_logits)
    body_flat = flatten_logits(body_logits)

    pred = np.zeros((3, H, W), dtype=np.float32)

    ph = head_flat.argmax(dim=1).item()
    pred[1, ph // W, ph % W] = 1.0

    pf = food_flat.argmax(dim=1).item()
    pred[2, pf // W, pf % W] = 1.0

    body_bin = (torch.sigmoid(body_flat) > 0.5).cpu().numpy().reshape(H, W)
    pred[0] = body_bin

    return pred


def evaluate_rollout(model, dataset, n_rollouts=ROLLOUT_N,
                     max_steps=ROLLOUT_MAX_STEPS, min_snake_length=ROLLOUT_MIN_SNAKE_LENGTH,
                     p_straight=ROLLOUT_P_STRAIGHT, seed=42):
    """
    Rollout evaluation: seed a rollout from a real dataset state, then drive
    both the model and the ground-truth simulator forward for up to max_steps
    using the same biased action sequence.

    Metrics computed:
        - Average rollout length before divergence (strict: first step where
          mean per-channel IoU drops below 0.9)
        - Per-step mean IoU at each step in ROLLOUT_DIVERGENCE_STEPS
        - Rollout invalid state rate (fraction of all rollout steps that
          produce an invalid predicted state)

    Starting states are filtered to have snake length >= min_snake_length to
    ensure rollouts have meaningful duration.

    The food channel is excluded from IoU on eating steps because food respawn
    is stochastic.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Filter dataset for valid, long-enough starting states
    valid_indices = []
    for idx in range(len(dataset)):
        state_hwc = dataset.states[idx]  # (H, W, C)
        body_cells = (state_hwc[:, :, 0] > 0.5).sum()
        head_cells = (state_hwc[:, :, 1] > 0.5).sum()
        snake_len = int(body_cells + head_cells)
        if snake_len >= min_snake_length and is_valid_state(
                torch.tensor(state_hwc).permute(2, 0, 1)
        ):
            valid_indices.append(idx)

    if len(valid_indices) < n_rollouts:
        print(f"  Warning: only {len(valid_indices)} valid starting states found, "
              f"requested {n_rollouts}.")
        n_rollouts = len(valid_indices)

    chosen = rng.choice(valid_indices, size=n_rollouts, replace=False).tolist()

    # Accumulators
    rollout_lengths = []  # steps before divergence
    step_ious = {s: [] for s in ROLLOUT_DIVERGENCE_STEPS}
    total_steps = 0
    total_invalid_steps = 0

    DIVERGENCE_THRESHOLD = 0.9

    for idx in chosen:
        state_hwc = dataset.states[idx].copy()  # (H, W, C)  ground-truth current

        # Infer current direction from head movement in the dataset transition
        # as the seed action; fall back to RIGHT if ambiguous.
        seed_action = int(dataset.actions[idx])
        current_action = seed_action

        model_state_chw = torch.tensor(state_hwc).permute(2, 0, 1).float()  # (C,H,W)
        gt_state_hwc = state_hwc.copy()

        diverged_at = max_steps  # pessimistic default

        for step in range(1, max_steps + 1):
            action = _biased_action(current_action, py_rng) if step > 1 else current_action
            current_action = action

            # --- Ground truth next state ---
            gt_next_hwc, gt_done = get_next_logical_frame(gt_state_hwc, action)
            if gt_next_hwc is None or gt_done:
                break  # terminal — end this rollout

            # Was food eaten this step?
            prev_food_pos = tuple(np.argwhere(gt_state_hwc[:, :, 2] == 1)[0][::-1])
            new_head_positions = np.argwhere(gt_next_hwc[:, :, 1] == 1)
            if len(new_head_positions) == 0:
                break
            new_head_pos = tuple(new_head_positions[0][::-1])
            ate_food = (new_head_pos == prev_food_pos)

            # --- Model prediction ---
            action_t = torch.tensor([action], dtype=torch.long).to(DEVICE)
            state_t = model_state_chw.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, _ = model(state_t, action_t)

            pred_chw = _decode_model_output(head_logits, body_logits, food_logits)

            # --- Validity ---
            total_steps += 1
            if not is_valid_state(pred_chw):
                total_invalid_steps += 1

            # --- IoU ---
            iou = _state_iou(pred_chw, gt_next_hwc, ate_food)

            if step in step_ious:
                step_ious[step].append(iou)

            # --- Divergence ---
            if iou < DIVERGENCE_THRESHOLD and diverged_at == max_steps:
                diverged_at = step

            # --- Advance ground truth; feed model its own prediction forward ---
            gt_state_hwc = gt_next_hwc
            # Convert predicted (C,H,W) back to (H,W,C) for the next model input
            model_state_chw = torch.tensor(pred_chw).float()

        rollout_lengths.append(diverged_at)

    # --- Report ---
    avg_length = float(np.mean(rollout_lengths)) if rollout_lengths else 0.0
    invalid_rate = total_invalid_steps / total_steps if total_steps > 0 else 0.0

    print(f"\n=== Rollout Evaluation (n={n_rollouts}, max_steps={max_steps}) ===")
    print(f"  Average rollout length before divergence: {avg_length:.1f} steps  "
          f"(threshold IoU < {DIVERGENCE_THRESHOLD})")
    print(f"  Rollout invalid state rate:           {invalid_rate:.4f}")
    print(f"  Per-step mean IoU:")
    for s in ROLLOUT_DIVERGENCE_STEPS:
        vals = step_ious[s]
        if vals:
            print(f"    step {s:>2}: {np.mean(vals):.4f}  (n={len(vals)})")
        else:
            print(f"    step {s:>2}: n/a  (no rollouts reached this step)")

    return {
        "avg_rollout_length": avg_length,
        "rollout_invalid_rate": invalid_rate,
        "step_ious": {s: float(np.mean(v)) if v else None
                      for s, v in step_ious.items()},
    }


# -------------------------
# Run All Models
# -------------------------
def eval_all_models():
    models = ["baseline", "transformer", "unet"]

    dataset = SnakeDataset(DATA_PATH)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Raw dataset needed for rollout (access .states / .actions directly)
    raw_test_dataset = test_dataset.dataset
    raw_test_indices = test_dataset.indices
    rollout_dataset = type('RolloutDataset', (), {
        'states': raw_test_dataset.states[raw_test_indices],
        'actions': raw_test_dataset.actions[raw_test_indices],
        '__len__': lambda self: len(self.states),
    })()

    for model_name in models:
        print("\n" + "=" * 50)
        print(f"Evaluating: {model_name.upper()}")
        print("=" * 50)

        model = load_model(model_name, DEVICE)

        evaluate_with_dataset(model, test_loader)
        evaluate_unseen(model, n_samples=2000)
        evaluate_rollout(model, rollout_dataset)


if __name__ == "__main__":
    eval_all_models()
