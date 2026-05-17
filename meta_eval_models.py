import numpy as np
import torch
import random

from select_model import load_model
from eval_tools import generate_random_snake_state, get_next_logical_frame, flatten_logits
from eval_models import (
    is_valid_state, _decode_model_output, _biased_action,
    GRID_SIZE, DEVICE, ACTIONS,
    ROLLOUT_N, ROLLOUT_MAX_STEPS, ROLLOUT_MIN_SNAKE_LENGTH,
    ROLLOUT_DIVERGENCE_STEPS, ROLLOUT_P_STRAIGHT,
)

# IoU threshold for a step to count as "survived"
SURVIVAL_IOU_THRESHOLD = 0.9


# Mean IoU across channels
def _state_iou_all_channels(pred_chw, gt_hwc, ate_food):
    channels = [0, 1] if ate_food else [0, 1, 2]
    ious = []
    for c in channels:
        pred = (pred_chw[c] > 0.5)
        gt = (gt_hwc[:, :, c] > 0.5)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        ious.append(1.0 if union == 0 else float(intersection) / float(union))
    return float(np.mean(ious))


# Meta Evaluation
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
    model.eval()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

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
            print(f"Warning: only {len(valid_indices)} valid starting states, "
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

    # Accumulators
    # Survival reward
    all_cumulative_survival = []  # one scalar per rollout
    step_survival = {s: [] for s in ROLLOUT_DIVERGENCE_STEPS}

    # Event detection F1 (reward events = food eaten)
    ed_tp = ed_fp = ed_fn = 0

    # Termination F1
    term_tp = term_fp = term_fn = 0

    # Rollout loop
    for i in range(n_rollouts):
        gt_state_hwc, current_action = get_start(i)
        model_state_chw = torch.tensor(gt_state_hwc.transpose(2, 0, 1)).float()

        cumulative_survival = 0.0

        for step in range(1, max_steps + 1):
            action = (_biased_action(current_action, py_rng) if step > 1 else current_action)
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

            # Survival reward
            iou = _state_iou_all_channels(pred_chw, gt_next_hwc, gt_event)
            survived = float(iou >= SURVIVAL_IOU_THRESHOLD)
            cumulative_survival += survived
            if step in step_survival:
                step_survival[step].append(survived)

            # Event detection F1
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

            # Termination F1
            if gt_done and pred_done:
                term_tp += 1
            elif not gt_done and pred_done:
                term_fp += 1
            elif gt_done and not pred_done:
                term_fn += 1

            # Ground truth moves forward, model feeds its own prediction
            gt_state_hwc = gt_next_hwc
            model_state_chw = torch.tensor(pred_chw).float()

            if gt_done:
                break

        all_cumulative_survival.append(cumulative_survival)

    # Compute final metrics

    # Survival reward
    mean_survival = float(np.mean(all_cumulative_survival))
    max_possible = max_steps  # theoretical ceiling

    # Event detection F1
    if (ed_tp + ed_fp) > 0:
        ed_prec = ed_tp / (ed_tp + ed_fp)
    else:
        ed_prec = 0.0
    if (ed_tp + ed_fn) > 0:
        ed_rec = ed_tp / (ed_tp + ed_fn)
    else:
        ed_rec = 0.0
    if (ed_prec + ed_rec) > 0:
        ed_f1 = 2 * ed_prec * ed_rec / (ed_prec + ed_rec)
    else:
        ed_f1 = 0.0
    ed_support = ed_tp + ed_fn

    # Termination F1
    if (term_tp + term_fp) > 0:
        term_prec = term_tp / (term_tp + term_fp)
    else:
        term_prec = 0.0
    if (term_tp + term_fn) > 0:
        term_rec = term_tp / (term_tp + term_fn)
    else:
        term_rec = 0.0
    if (term_prec + term_rec) > 0:
        term_f1 = 2 * term_prec * term_rec / (term_prec + term_rec)
    else:
        term_f1 = 0.0

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


# Run all models
def meta_eval_all_models():
    from torch.utils.data import random_split
    from eval_models import SnakeDataset, BATCH_SIZE
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
