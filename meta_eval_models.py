"""
meta_eval_model.py
==================
Game-agnostic evaluation of the neural Snake world model.

Metrics are framed to be applicable to any neural world model, not Snake
specifically. All three metrics are computed over paired rollouts — the same
starting state and action sequence is run through both the model and the
ground-truth simulator, and the results are compared.

Metrics
-------
1. Survival Reward
   +1 for each rollout step where the predicted state has IoU >= threshold
   against the ground-truth next state. Measures how long the model produces
   correct next states before diverging. Reported as mean cumulative survival
   reward across all rollouts, and as a survival reward curve (mean per step).

2. Event Detection F1
   Binary F1 on whether the model correctly detects discrete reward events
   (food eaten in Snake; generalises to any game event). Positive class =
   event occurred this step. Accumulates TP/FP/FN across all rollout steps.

3. Termination F1
   Binary F1 on whether the model correctly predicts episode termination.
   Positive class = episode ends this step. Ground truth is the simulator
   done flag; prediction is the model's done head output.
"""

import numpy as np
import torch
import random

from select_model import load_model
from eval_tools import generate_random_snake_state, get_next_logical_frame
from eval_model import (
    is_valid_state, flatten_logits, _decode_model_output, _biased_action,
    rules_based_next_state,
    GRID_SIZE, DEVICE, ACTIONS,
    ROLLOUT_N, ROLLOUT_MAX_STEPS, ROLLOUT_MIN_SNAKE_LENGTH,
    ROLLOUT_DIVERGENCE_STEPS, ROLLOUT_P_STRAIGHT,
)

# IoU threshold for a step to count as "survived"
SURVIVAL_IOU_THRESHOLD = 0.9


# -------------------------
# IoU helper
# -------------------------
def _state_iou_all_channels(pred_chw, gt_hwc, ate_food):
    """
    Mean IoU across channels. Food channel excluded on eating steps
    since food respawn is stochastic.
    pred_chw : (C, H, W)  |  gt_hwc : (H, W, C)
    """
    channels = [0, 1] if ate_food else [0, 1, 2]
    ious = []
    for c in channels:
        pred = (pred_chw[c] > 0.5)
        gt = (gt_hwc[:, :, c] > 0.5)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        ious.append(1.0 if union == 0 else float(intersection) / float(union))
    return float(np.mean(ious))


# -------------------------
# Meta Evaluation
# -------------------------
def meta_evaluate(
        model,
        model_name="model",
        n_rollouts=ROLLOUT_N,
        max_steps=ROLLOUT_MAX_STEPS,
        min_snake_length=ROLLOUT_MIN_SNAKE_LENGTH,
        grid_size=GRID_SIZE,
        dataset=None,
        seed=42,
):
    """
    Runs paired rollouts and computes the three meta metrics.

    Starting states are sampled from the dataset when grid_size matches the
    training size and a dataset is provided; otherwise generated procedurally.

    Args:
        model           : loaded, eval-mode model
        model_name      : string label for printing
        n_rollouts      : number of rollouts to run
        max_steps       : maximum steps per rollout
        min_snake_length: minimum snake length for starting states
        grid_size       : grid size for state generation / decoding
        dataset         : optional RolloutDataset (states, actions arrays)
        seed            : random seed for reproducibility
    """
    model.eval()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    use_dataset = (grid_size == GRID_SIZE) and (dataset is not None)

    # --- Build starting-state provider ---
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
            print(f"  Warning: only {len(valid_indices)} valid starting states, "
                  f"requested {n_rollouts}.")
            n_rollouts = len(valid_indices)

        chosen = rng.choice(valid_indices, size=n_rollouts, replace=False).tolist()

        def get_start(i):
            hwc = dataset.states[chosen[i]].copy()
            action = int(dataset.actions[chosen[i]])
            return hwc, action
    else:
        def get_start(_):
            state_chw, _, _ = generate_random_snake_state(grid_size)
            return state_chw.transpose(1, 2, 0), random.choice(ACTIONS)

    # --- Accumulators ---
    # Survival reward
    all_cumulative_survival = []  # one scalar per rollout
    step_survival = {s: [] for s in ROLLOUT_DIVERGENCE_STEPS}

    # Event detection F1 (reward events = food eaten)
    ed_tp = ed_fp = ed_fn = 0

    # Termination F1
    term_tp = term_fp = term_fn = 0

    # --- Rollout loop ---
    for i in range(n_rollouts):
        gt_state_hwc, current_action = get_start(i)
        model_state_chw = torch.tensor(
            gt_state_hwc.transpose(2, 0, 1)
        ).float()

        cumulative_survival = 0.0

        for step in range(1, max_steps + 1):
            action = (_biased_action(current_action, py_rng)
                      if step > 1 else current_action)
            current_action = action

            # Ground truth step
            gt_next_hwc, gt_done = get_next_logical_frame(gt_state_hwc, action)
            if gt_next_hwc is None:
                break

            # Detect food eaten (reward event)
            prev_food_pos = tuple(np.argwhere(gt_state_hwc[:, :, 2] == 1)[0][::-1])
            new_head_arr = np.argwhere(gt_next_hwc[:, :, 1] == 1)
            if len(new_head_arr) == 0:
                break
            gt_event = (tuple(new_head_arr[0][::-1]) == prev_food_pos)

            # Model prediction
            action_t = torch.tensor([action], dtype=torch.long).to(DEVICE)
            state_t = model_state_chw.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, done_logit = model(
                    state_t, action_t
                )

            pred_chw = _decode_model_output(
                head_logits, body_logits, food_logits, grid_size
            )
            pred_done = (torch.sigmoid(done_logit).item() > 0.5)

            # ---- Survival reward ----
            iou = _state_iou_all_channels(pred_chw, gt_next_hwc, gt_event)
            survived = float(iou >= SURVIVAL_IOU_THRESHOLD)
            cumulative_survival += survived
            if step in step_survival:
                step_survival[step].append(survived)

            # ---- Event detection F1 ----
            # Predicted event: model predicts head on current food cell
            prev_food_flat = torch.tensor(
                gt_state_hwc[:, :, 2], dtype=torch.float32
            ).view(-1)
            prev_food_idx = int(prev_food_flat.argmax().item())
            pred_head_flat = flatten_logits(head_logits)
            pred_head_idx = int(pred_head_flat.argmax(dim=1).item())
            pred_event = (pred_head_idx == prev_food_idx)

            if gt_event and pred_event:
                ed_tp += 1
            elif not gt_event and pred_event:
                ed_fp += 1
            elif gt_event and not pred_event:
                ed_fn += 1

            # ---- Termination F1 ----
            if gt_done and pred_done:
                term_tp += 1
            elif not gt_done and pred_done:
                term_fp += 1
            elif gt_done and not pred_done:
                term_fn += 1

            # Advance
            gt_state_hwc = gt_next_hwc
            model_state_chw = torch.tensor(pred_chw).float()

            if gt_done:
                break

        all_cumulative_survival.append(cumulative_survival)

    # --- Compute final metrics ---

    # Survival reward
    mean_survival = float(np.mean(all_cumulative_survival))
    max_possible = max_steps  # theoretical ceiling

    # Event detection F1
    ed_prec = ed_tp / (ed_tp + ed_fp) if (ed_tp + ed_fp) > 0 else 0.0
    ed_rec = ed_tp / (ed_tp + ed_fn) if (ed_tp + ed_fn) > 0 else 0.0
    ed_f1 = (2 * ed_prec * ed_rec / (ed_prec + ed_rec)
             if (ed_prec + ed_rec) > 0 else 0.0)
    ed_support = ed_tp + ed_fn

    # Termination F1
    term_prec = term_tp / (term_tp + term_fp) if (term_tp + term_fp) > 0 else 0.0
    term_rec = term_tp / (term_tp + term_fn) if (term_tp + term_fn) > 0 else 0.0
    term_f1 = (2 * term_prec * term_rec / (term_prec + term_rec)
               if (term_prec + term_rec) > 0 else 0.0)
    term_support = term_tp + term_fn

    metrics = {
        "mean_survival_reward": mean_survival,
        "max_possible_survival": max_possible,
        "survival_rate": mean_survival / max_possible,
        "step_survival": {s: float(np.mean(v)) if v else None
                          for s, v in step_survival.items()},
        "event_detection_f1": ed_f1,
        "event_detection_prec": ed_prec,
        "event_detection_rec": ed_rec,
        "event_support": ed_support,
        "termination_f1": term_f1,
        "termination_prec": term_prec,
        "termination_rec": term_rec,
        "termination_support": term_support,
    }

    _print_meta_metrics(metrics, model_name, grid_size, n_rollouts, max_steps)
    return metrics


def _print_meta_metrics(metrics, model_name, grid_size, n_rollouts, max_steps):
    print(f"\n=== Meta Evaluation: {model_name.upper()} "
          f"(grid={grid_size}, n={n_rollouts}, max_steps={max_steps}) ===")

    print(f"\n  -- Survival Reward (IoU threshold={SURVIVAL_IOU_THRESHOLD}) --")
    print(f"  Mean cumulative survival:  {metrics['mean_survival_reward']:.2f} "
          f"/ {metrics['max_possible_survival']}  "
          f"({metrics['survival_rate'] * 100:.1f}%)")
    print(f"  Per-step survival rate:")
    for s, v in metrics["step_survival"].items():
        if v is not None:
            print(f"    step {s:>2}: {v:.4f}")
        else:
            print(f"    step {s:>2}: n/a")

    print(f"\n  -- Event Detection F1 (support={metrics['event_support']}) --")
    print(f"  F1:        {metrics['event_detection_f1']:.4f}")
    print(f"  Precision: {metrics['event_detection_prec']:.4f}")
    print(f"  Recall:    {metrics['event_detection_rec']:.4f}")

    print(f"\n  -- Termination F1 (support={metrics['termination_support']}) --")
    print(f"  F1:        {metrics['termination_f1']:.4f}")
    print(f"  Precision: {metrics['termination_prec']:.4f}")
    print(f"  Recall:    {metrics['termination_rec']:.4f}")


# -------------------------
# Run All Models
# -------------------------
def meta_eval_all_models():
    from torch.utils.data import random_split
    from eval_model import SnakeDataset, BATCH_SIZE
    from torch.utils.data import DataLoader

    models = ["baseline", "transformer", "unet"]

    dataset = SnakeDataset("snake_transitions.npz")
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    raw = test_dataset.dataset
    indices = test_dataset.indices
    rollout_ds = type("RolloutDataset", (), {
        "states": raw.states[indices],
        "actions": raw.actions[indices],
        "__len__": lambda self: len(self.states),
    })()

    all_results = {}

    for model_name in models:
        print("\n" + "=" * 50)
        print(f"Meta Evaluating: {model_name.upper()}")
        print("=" * 50)

        model = load_model(model_name, DEVICE)

        metrics = meta_evaluate(
            model,
            model_name=model_name,
            dataset=rollout_ds,
            grid_size=GRID_SIZE,
        )
        all_results[model_name] = metrics

    _print_comparison_table(all_results, models)
    return all_results


def _print_comparison_table(all_results, models):
    print("\n" + "=" * 70)
    print("META EVALUATION SUMMARY")
    print("=" * 70)

    header = (f"{'Model':>12}  {'SurvivalRate':>13}  "
              f"{'EventF1':>8}  {'TermF1':>7}")
    print(header)
    print("─" * 70)

    for model_name in models:
        m = all_results.get(model_name, {})
        print(
            f"{model_name:>12}  "
            f"{m.get('survival_rate', float('nan')):>13.4f}  "
            f"{m.get('event_detection_f1', float('nan')):>8.4f}  "
            f"{m.get('termination_f1', float('nan')):>7.4f}"
        )

    print("─" * 70)


if __name__ == "__main__":
    meta_eval_all_models()
