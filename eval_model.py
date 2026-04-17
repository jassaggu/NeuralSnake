import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import deque
import random

from select_model import load_model
from eval_tools import generate_random_snake_state

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
        all_food_correct,
        all_done_correct,
        all_body_tp, all_body_fp, all_body_fn,
        all_food_consumption_correct, all_food_consumption_total,
        all_illegal,
        total,
):
    head_acc = sum(all_head_correct) / total
    food_acc = sum(all_food_correct) / total
    done_acc = sum(all_done_correct) / total

    tp = sum(all_body_tp)
    fp = sum(all_body_fp)
    fn = sum(all_body_fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    fc_total = sum(all_food_consumption_total)
    food_consumption_acc = (sum(all_food_consumption_correct) / fc_total
                            if fc_total > 0 else float("nan"))

    illegal_rate = sum(all_illegal) / total

    return {
        "head_acc": head_acc,
        "food_acc": food_acc,
        "done_acc": done_acc,
        "body_f1": f1,
        "food_consumption_acc": food_consumption_acc,
        "illegal_rate": illegal_rate,
    }


def print_metrics(metrics, label):
    print(f"\n=== {label} ===")
    print(f"  Head Accuracy:           {metrics['head_acc']:.4f}")
    print(f"  Food Position Accuracy:  {metrics['food_acc']:.4f}")
    print(f"  Done Accuracy:           {metrics['done_acc']:.4f}")
    print(f"  Body F1 Score:           {metrics['body_f1']:.4f}")
    print(f"  Food Consumption Acc:    {metrics['food_consumption_acc']:.4f}")
    print(f"  Illegal State Rate:      {metrics['illegal_rate']:.4f}")


def _accumulate_batch(
        states, actions, next_states, dones, model,
        all_head_correct, all_food_correct, all_done_correct,
        all_body_tp, all_body_fp, all_body_fn,
        all_food_consumption_correct, all_food_consumption_total,
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

    # Food position accuracy
    all_food_correct.extend((pred_food_idx == gt_food_idx).cpu().tolist())

    # Done accuracy
    all_done_correct.extend((pred_done.squeeze(-1) == dones).cpu().tolist())

    # Body F1 components
    tp = (pred_body_bin * target_body).sum(dim=1)
    fp = (pred_body_bin * (1 - target_body)).sum(dim=1)
    fn = ((1 - pred_body_bin) * target_body).sum(dim=1)
    all_body_tp.extend(tp.cpu().tolist())
    all_body_fp.extend(fp.cpu().tolist())
    all_body_fn.extend(fn.cpu().tolist())

    # Food consumption accuracy
    # A food-consumption event occurs when the head moves onto the food cell.
    prev_food = states[:, 2].view(B, -1)  # food in current state
    gt_consumed = (gt_head_idx == prev_food.argmax(dim=1))
    pred_consumed = (pred_head_idx == prev_food.argmax(dim=1))
    for gt_c, pr_c in zip(gt_consumed.cpu().tolist(), pred_consumed.cpu().tolist()):
        if gt_c:  # only score on actual consumption events
            all_food_consumption_correct.append(int(gt_c == pr_c))
            all_food_consumption_total.append(1)

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
        "head", "food", "done", "body_tp", "body_fp", "body_fn",
        "fc_correct", "fc_total", "illegal"
    ]}

    for states, actions, next_states, dones in loader:
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)

        _accumulate_batch(
            states, actions, next_states, dones, model,
            acc["head"], acc["food"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_correct"], acc["fc_total"],
            acc["illegal"],
        )

    total = len(acc["head"])
    metrics = compute_metrics(
        acc["head"], acc["food"], acc["done"],
        acc["body_tp"], acc["body_fp"], acc["body_fn"],
        acc["fc_correct"], acc["fc_total"],
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
        "head", "food", "done", "body_tp", "body_fp", "body_fn",
        "fc_correct", "fc_total", "illegal"
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
            acc["head"], acc["food"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_correct"], acc["fc_total"],
            acc["illegal"],
        )

    if skipped:
        print(f"  (skipped {skipped} samples due to generation failures)")

    total = len(acc["head"])
    if total == 0:
        print("  No samples evaluated.")
        return {}

    metrics = compute_metrics(
        acc["head"], acc["food"], acc["done"],
        acc["body_tp"], acc["body_fp"], acc["body_fn"],
        acc["fc_correct"], acc["fc_total"],
        acc["illegal"], total,
    )
    print_metrics(metrics, "Unseen State Evaluation (procedurally generated)")
    return metrics


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

    for model_name in models:
        print("\n" + "=" * 50)
        print(f"Evaluating: {model_name.upper()}")
        print("=" * 50)

        model = load_model(model_name, DEVICE)

        evaluate_with_dataset(model, test_loader)
        evaluate_unseen(model, n_samples=2000)


if __name__ == "__main__":
    eval_all_models()
