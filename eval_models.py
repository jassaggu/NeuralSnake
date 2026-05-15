import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import deque
import random

from select_model import load_model
from eval_tools import get_next_logical_frame, generate_random_snake_state, flatten_logits, rules_based_next_state

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "snake_transitions.npz"
BATCH_SIZE = 128
GRID_SIZE = 10  # Default/native is 10, can change freely

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Rollout config
ROLLOUT_DIVERGENCE_STEPS = [1, 5, 10, 20, 30, 50]
ROLLOUT_MAX_STEPS = 50
ROLLOUT_N = 100
ROLLOUT_MIN_SNAKE_LENGTH = 4
ROLLOUT_P_STRAIGHT = 0.6  # Pro
ROLLOUT_DIVERGENCE_THRESHOLD = 0.9


# Dataset
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


# State Validity Check
def is_valid_state(state_tensor):
    # Checks whether a state represents a legal Snake game state:
    # Exactly one head and food cell
    # No overlap between any pair of channels
    # All body cells connected to head

    if isinstance(state_tensor, torch.Tensor):
        state = state_tensor.cpu().numpy()
    else:
        state = state_tensor

    body = (state[0] > 0.5).astype(np.uint8)
    head = (state[1] > 0.5).astype(np.uint8)
    food = (state[2] > 0.5).astype(np.uint8)

    if head.sum() != 1:
        return False
    if food.sum() != 1:
        return False
    if np.any((head + body) > 1):
        return False
    if np.any((head + food) > 1):
        return False
    if np.any((body + food) > 1):
        return False

    if int(body.sum()) > 0:
        H, W = head.shape
        head_pos = tuple(np.argwhere(head == 1)[0])
        occupied = (head | body).astype(bool)
        visited = {head_pos}
        queue = deque([head_pos])

        while queue:
            current_row, current_col = queue.popleft()

            # Neighbour cells
            for row_offset, col_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbour_row = current_row + row_offset
                neighbour_col = current_col + col_offset

                if 0 <= neighbour_row < H and 0 <= neighbour_col < W:
                    if occupied[neighbour_row, neighbour_col] and (neighbour_row, neighbour_col) not in visited:
                        visited.add((neighbour_row, neighbour_col))
                        queue.append((neighbour_row, neighbour_col))

        body_positions = set(map(tuple, np.argwhere(body == 1)))
        if not body_positions.issubset(visited):
            return False

    return True


# Compute chosen metrics
def compute_metrics(
        all_head_correct, all_done_correct,
        all_body_tp, all_body_fp, all_body_fn,
        all_fc_tp, all_fc_fp, all_fc_fn,
        all_food_static_correct, all_food_static_total,
        all_food_respawn_correct, all_food_respawn_total,
        all_illegal,
        total,
):
    # Head and done accuracy
    head_acc = sum(all_head_correct) / total
    done_acc = sum(all_done_correct) / total

    # Body prediction metrics
    tp = sum(all_body_tp)
    fp = sum(all_body_fp)
    fn = sum(all_body_fn)

    # Calculate precision
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0.0
    # Calculate recall
    if (tp + fn) > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0.0
    # F1 score
    if (prec + rec) > 0:
        body_f1 = 2 * prec * rec / (prec + rec)
    else:
        body_f1 = 0.0

    # Food consumption metrics
    fc_tp = sum(all_fc_tp)
    fc_fp = sum(all_fc_fp)
    fc_fn = sum(all_fc_fn)

    # Precision
    if (fc_tp + fc_fp) > 0:
        fc_prec = fc_tp / (fc_tp + fc_fp)
    else:
        fc_prec = 0.0
    # Recall
    if (fc_tp + fc_fn) > 0:
        fc_rec = fc_tp / (fc_tp + fc_fn)
    else:
        fc_rec = 0.0
    # F1 score
    if (fc_prec + fc_rec) > 0:
        fc_f1 = 2 * fc_prec * fc_rec / (fc_prec + fc_rec)
    else:
        fc_f1 = 0.0

    fc_support = fc_tp + fc_fn

    # Food static accuracy
    fs_total = sum(all_food_static_total)
    if fs_total > 0:
        food_static_acc = sum(all_food_static_correct) / fs_total
    else:
        food_static_acc = float("nan")

    # Food respawn accuracy
    fr_total = sum(all_food_respawn_total)
    if fr_total > 0:
        food_respawn_acc = (sum(all_food_respawn_correct) / fr_total)
    else:
        food_respawn_acc = float("nan")

    # Illegal state rate
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
    print(f"  Food Consumption F1:        {metrics['fc_f1']:.4f}  "
          f"(support: {metrics['fc_support']})")
    print(f"  Food Static Accuracy:       {metrics['food_static_acc']:.4f}  "
          f"(n={metrics['food_static_n']})")
    print(f"  Food Respawn Accuracy:      {metrics['food_respawn_acc']:.4f}  "
          f"(n={metrics['food_respawn_n']})")
    print(f"  Illegal State Rate:         {metrics['illegal_rate']:.4f}")


# Central function that runs a batch through the model and accumulates metric lists
def accumulate_batch(
        states, actions, next_states, dones, model,
        all_head_correct, all_done_correct,
        all_body_tp, all_body_fp, all_body_fn,
        all_fc_tp, all_fc_fp, all_fc_fn,
        all_food_static_correct, all_food_static_total,
        all_food_respawn_correct, all_food_respawn_total,
        all_illegal,
        grid_size=GRID_SIZE,
):
    with torch.no_grad():
        head_logits, body_logits, food_logits, done_logit = model(states, actions)

    B = states.size(0)
    H = W = grid_size

    head_flat = flatten_logits(head_logits)
    food_flat = flatten_logits(food_logits)
    body_flat = flatten_logits(body_logits)

    pred_head_idx = head_flat.argmax(dim=1)
    pred_food_idx = food_flat.argmax(dim=1)
    pred_body_bin = (torch.sigmoid(body_flat) > 0.5).float()
    pred_done = (torch.sigmoid(done_logit) > 0.5).float()

    target_head = next_states[:, 1].view(B, -1)
    target_food = next_states[:, 2].view(B, -1)
    target_body = next_states[:, 0].view(B, -1)

    gt_head_idx = target_head.argmax(dim=1)
    gt_food_idx = target_food.argmax(dim=1)

    # Head accuracy
    all_head_correct.extend((pred_head_idx == gt_head_idx).cpu().tolist())

    # Done accuracy
    all_done_correct.extend((pred_done.squeeze(-1) == dones).cpu().tolist())

    # Body F1
    tp = (pred_body_bin * target_body).sum(dim=1)
    fp = (pred_body_bin * (1 - target_body)).sum(dim=1)
    fn = ((1 - pred_body_bin) * target_body).sum(dim=1)
    all_body_tp.extend(tp.cpu().tolist())
    all_body_fp.extend(fp.cpu().tolist())
    all_body_fn.extend(fn.cpu().tolist())

    # Food consumption F1
    prev_food_idx = states[:, 2].view(B, -1).argmax(dim=1)
    gt_consumed = (gt_head_idx == prev_food_idx)
    pred_consumed = (pred_head_idx == prev_food_idx)

    all_fc_tp.extend((gt_consumed & pred_consumed).long().cpu().tolist())
    all_fc_fp.extend((~gt_consumed & pred_consumed).long().cpu().tolist())
    all_fc_fn.extend((gt_consumed & ~pred_consumed).long().cpu().tolist())

    # Food static accuracy (non eating steps)
    static_mask = ~ gt_consumed
    if static_mask.any():
        correct = (pred_food_idx[static_mask] == gt_food_idx[static_mask])
        all_food_static_correct.extend(correct.cpu().tolist())
        all_food_static_total.extend([1] * int(static_mask.sum().item()))

    # Food respawn accuracy (eating steps)
    if gt_consumed.any():
        correct = (pred_food_idx[gt_consumed] == gt_food_idx[gt_consumed])
        all_food_respawn_correct.extend(correct.cpu().tolist())
        all_food_respawn_total.extend([1] * int(gt_consumed.sum().item()))

    # Illegal state rate
    pred_head_np = pred_head_idx.cpu().numpy()
    pred_food_np = pred_food_idx.cpu().numpy()
    pred_body_np = pred_body_bin.cpu().numpy()

    for i in range(B):
        pred_state = np.zeros((3, H, W), dtype=np.float32)
        ph = pred_head_np[i]
        pred_state[1, ph // W, ph % W] = 1.0
        pf = pred_food_np[i]
        pred_state[2, pf // W, pf % W] = 1.0
        pred_state[0] = pred_body_np[i].reshape(H, W)
        all_illegal.append(0 if is_valid_state(pred_state) else 1)


# Evaluation 1 - Dataset
def evaluate_with_dataset(model, loader, grid_size=GRID_SIZE):
    model.eval()

    # Make lists for metrics
    acc = {k: [] for k in [
        "head", "done", "body_tp", "body_fp", "body_fn",
        "fc_tp", "fc_fp", "fc_fn",
        "food_static_correct", "food_static_total",
        "food_respawn_correct", "food_respawn_total",
        "illegal",
    ]}

    for states, actions, next_states, dones in loader:
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)

        accumulate_batch(
            states, actions, next_states, dones, model,
            acc["head"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
            acc["food_static_correct"], acc["food_static_total"],
            acc["food_respawn_correct"], acc["food_respawn_total"],
            acc["illegal"],
            grid_size=grid_size,
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
    print_metrics(metrics, f"Dataset Evaluation (grid={grid_size}, held-out test split)")
    return metrics


# Evaluation 2 - Unseen
def evaluate_unseen(model, n_samples=2000, grid_size=GRID_SIZE):
    model.eval()

    acc = {k: [] for k in [
        "head", "done", "body_tp", "body_fp", "body_fn",
        "fc_tp", "fc_fp", "fc_fn",
        "food_static_correct", "food_static_total",
        "food_respawn_correct", "food_respawn_total",
        "illegal",
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

        state_t = torch.tensor(state_np).unsqueeze(0).to(DEVICE)
        next_state_t = torch.tensor(next_state_np).unsqueeze(0).to(DEVICE)
        action_t = torch.tensor([action], dtype=torch.long).to(DEVICE)
        done_t = torch.tensor([float(done)]).to(DEVICE)

        accumulate_batch(
            state_t, action_t, next_state_t, done_t, model,
            acc["head"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
            acc["food_static_correct"], acc["food_static_total"],
            acc["food_respawn_correct"], acc["food_respawn_total"],
            acc["illegal"],
            grid_size=grid_size,
        )

    if skipped:
        print(f"(skipped {skipped} samples due to generation failures)")

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
    print_metrics(metrics, f"Unseen Evaluation (grid={grid_size}, procedurally generated)")

    return metrics


# Rollout Helpers

# Choose next action with bias for going straight to prevent overcongestion, uses ROLLOUT_P_STRAIGHT
def _biased_action(current_action, rng):
    opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

    turn_options = []
    for a in ACTIONS:
        if a != opposite[current_action] and a != current_action:
            turn_options.append(a)

    if rng.random() < ROLLOUT_P_STRAIGHT:
        return current_action

    return rng.choice(turn_options)


# Copmute IoU between two grid channels
def _channel_iou(pred_channel, gt_channel):
    pred = pred_channel > 0.5
    gt = gt_channel > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return float(intersection) / float(union)


# Computes mean IoU across relevant channels of the Snake state
# Includes body/head always, and food only if not eaten
def _state_iou(pred_chw, gt_hwc, ate_food):
    channels = [0, 1] if ate_food else [0, 1, 2]

    ious = []
    for c in channels:
        ious.append(_channel_iou(pred_chw[c], gt_hwc[:, :, c]))

    return float(np.mean(ious))


# Converts model logits into a Snake state grid (3xHxW)
def _decode_model_output(head_logits, body_logits, food_logits, grid_size):
    H = W = grid_size
    head_flat = flatten_logits(head_logits)
    food_flat = flatten_logits(food_logits)
    body_flat = flatten_logits(body_logits)

    pred = np.zeros((3, H, W), dtype=np.float32)

    ph = head_flat.argmax(dim=1).item()
    pred[1, ph // W, ph % W] = 1.0

    pf = food_flat.argmax(dim=1).item()
    pred[2, pf // W, pf % W] = 1.0

    pred[0] = (torch.sigmoid(body_flat) > 0.5).cpu().numpy().reshape(H, W)

    return pred


# Evaluation 3 - Rollout
def evaluate_rollout(model, dataset=None, n_rollouts=ROLLOUT_N, max_steps=ROLLOUT_MAX_STEPS,
                     min_snake_length=ROLLOUT_MIN_SNAKE_LENGTH, grid_size=GRID_SIZE, seed=42):
    model.eval()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Uses a dataset state to start rollout if grid size is native, otherwise procedurally generates state
    use_dataset = (grid_size == GRID_SIZE) and (dataset is not None)

    if use_dataset:
        valid_indices = []
        for idx in range(len(dataset)):
            state_hwc = dataset.states[idx]
            snake_len = int(
                (state_hwc[:, :, 0] > 0.5).sum() +
                (state_hwc[:, :, 1] > 0.5).sum()
            )
            if snake_len >= min_snake_length and is_valid_state(
                    torch.tensor(state_hwc).permute(2, 0, 1)
            ):
                valid_indices.append(idx)

        if len(valid_indices) < n_rollouts:
            print(f"Warning: only {len(valid_indices)} valid starting states "
                  f"found, requested {n_rollouts}.")
            n_rollouts = len(valid_indices)

        chosen = rng.choice(valid_indices, size=n_rollouts, replace=False).tolist()

        def get_start(i):
            hwc = dataset.states[chosen[i]].copy()
            action = int(dataset.actions[chosen[i]])
            return hwc, action
    else:
        def get_start(_):
            state_chw, _, _ = generate_random_snake_state(grid_size)
            hwc = state_chw.transpose(1, 2, 0)
            return hwc, random.choice(ACTIONS)

    # Rollout loop
    rollout_lengths = []
    step_ious = {s: [] for s in ROLLOUT_DIVERGENCE_STEPS}
    total_steps = 0
    total_invalid_steps = 0

    for i in range(n_rollouts):
        gt_state_hwc, current_action = get_start(i)
        model_state_chw = torch.tensor(gt_state_hwc.transpose(2, 0, 1)).float()

        diverged_at = max_steps

        for step in range(1, max_steps + 1):
            action = (_biased_action(current_action, py_rng) if step > 1 else current_action)
            current_action = action

            # Get the next ground truth state
            gt_next_hwc, gt_done = get_next_logical_frame(gt_state_hwc, action)
            if gt_next_hwc is None or gt_done:
                break

            # Detect food eaten this step
            prev_food_pos = tuple(np.argwhere(gt_state_hwc[:, :, 2] == 1)[0][::-1])
            new_head_arr = np.argwhere(gt_next_hwc[:, :, 1] == 1)
            if len(new_head_arr) == 0:
                break
            ate_food = (tuple(new_head_arr[0][::-1]) == prev_food_pos)

            # Model prediction
            action_t = torch.tensor([action], dtype=torch.long).to(DEVICE)
            state_t = model_state_chw.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, _ = model(state_t, action_t)

            pred_chw = _decode_model_output(head_logits, body_logits, food_logits, grid_size)

            # Validity
            total_steps += 1
            if not is_valid_state(pred_chw):
                total_invalid_steps += 1

            # IoU
            iou = _state_iou(pred_chw, gt_next_hwc, ate_food)
            if step in step_ious:
                step_ious[step].append(iou)

            # If divergence occurs (according to our defined threshold)
            if iou < ROLLOUT_DIVERGENCE_THRESHOLD and diverged_at == max_steps:
                diverged_at = step

            # Ground truth moves forward, model feeds its own prediction
            gt_state_hwc = gt_next_hwc
            model_state_chw = torch.tensor(pred_chw).float()

        rollout_lengths.append(diverged_at)

    avg_length = float(np.mean(rollout_lengths)) if rollout_lengths else 0.0
    invalid_rate = total_invalid_steps / total_steps if total_steps > 0 else 0.0

    print(f"\n=== Rollout Evaluation "
          f"(grid={grid_size}, n={n_rollouts}, max_steps={max_steps}) ===")
    print(f"  Avg steps before divergence:  {avg_length:.1f}  "
          f"(IoU threshold={ROLLOUT_DIVERGENCE_THRESHOLD})")
    print(f"  Rollout invalid state rate:   {invalid_rate:.4f}")
    print(f"  Per-step mean IoU:")
    for s in ROLLOUT_DIVERGENCE_STEPS:
        vals = step_ious[s]
        if vals:
            print(f"    step {s:>2}: {np.mean(vals):.4f}  (n={len(vals)})")
        else:
            print(f"    step {s:>2}: n/a")

    return {
        "avg_rollout_length": avg_length,
        "rollout_invalid_rate": invalid_rate,
        "step_ious": {s: float(np.mean(v)) if v else None
                      for s, v in step_ious.items()},
    }


# Run all models
def eval_all_models():
    models = ["baseline", "transformer", "unet"]

    dataset = SnakeDataset(DATA_PATH)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get raw arrays for rollout
    raw = test_dataset.dataset
    indices = test_dataset.indices
    rollout_ds = type("RolloutDataset", (), {
        "states": raw.states[indices],
        "actions": raw.actions[indices],
        "__len__": lambda self: len(self.states),
    })()

    for model_name in models:
        print("\n" + "=" * 50)
        print(f"Evaluating: {model_name.upper()}")
        print("=" * 50)

        model = load_model(model_name, DEVICE)

        evaluate_with_dataset(model, test_loader, grid_size=GRID_SIZE)
        evaluate_unseen(model, n_samples=2000, grid_size=GRID_SIZE)
        evaluate_rollout(model, rollout_ds, grid_size=GRID_SIZE)


if __name__ == "__main__":
    eval_all_models()
