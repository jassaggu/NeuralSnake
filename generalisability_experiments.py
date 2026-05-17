import torch
import random

from select_model import load_model
from eval_tools import generate_random_snake_state, rules_based_next_state
from eval_models import (
    evaluate_unseen, evaluate_rollout,
    accumulate_batch, compute_metrics, print_metrics,
    GRID_SIZE, DEVICE, ACTIONS,
)

# Dataset statistics: mean=13.07, std=8.03, min=1, max=47
LENGTH_BINS = {
    "short": (1, 6),
    "medium": (7, 19),
    "long": (20, 35),
    "very_long": (36, 47),
}


# Experiment 1 - grid size generalisation
# Evaluates the model on grid sizes from 6x6 to 15x15.
def grid_size_generalisation(
        model_name="unet",
        grid_sizes=range(6, 20),
        n_unseen_samples=2000,
        n_rollouts=100,
):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Grid Size Generalisation")
    print(f"  Model:       {model_name.upper()}")
    print(f"  Train size:  {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Test sizes:  {list(grid_sizes)}")
    print("=" * 60)

    model = load_model(model_name, DEVICE)

    all_results = {}

    for gs in grid_sizes:
        marker = " <-- train size" if gs == GRID_SIZE else ""
        print(f"\n{'─' * 50}")
        print(f"  Grid size: {gs}x{gs}{marker}")
        print(f"{'─' * 50}")

        unseen_metrics = evaluate_unseen(
            model,
            n_samples=n_unseen_samples,
            grid_size=gs,
        )

        rollout_metrics = evaluate_rollout(
            model,
            dataset=None,
            n_rollouts=n_rollouts,
            grid_size=gs,
        )

        all_results[gs] = {
            "unseen": unseen_metrics,
            "rollout": rollout_metrics,
        }

    _print_grid_size_summary(all_results, grid_sizes)
    return all_results


def _print_grid_size_summary(all_results, grid_sizes):
    print("\n" + "=" * 80)
    print("SUMMARY — Grid Size Generalisation")
    print("=" * 80)

    header = (f"{'Grid':>6}  {'HeadAcc':>8}  {'BodyF1':>7}  {'FC_F1':>6}  "
              f"{'Illegal':>8}  {'RolloutLen':>11}  {'RolloutInv':>11}")
    print(header)
    print("─" * 80)

    for gs in grid_sizes:
        r = all_results.get(gs)
        if r is None:
            continue
        u = r["unseen"]
        ro = r["rollout"]
        marker = "*" if gs == GRID_SIZE else " "
        print(
            f"{gs:>5}{marker}  "
            f"{u.get('head_acc', float('nan')):>8.4f}  "
            f"{u.get('body_f1', float('nan')):>7.4f}  "
            f"{u.get('fc_f1', float('nan')):>6.4f}  "
            f"{u.get('illegal_rate', float('nan')):>8.4f}  "
            f"{ro.get('avg_rollout_length', float('nan')):>11.1f}  "
            f"{ro.get('rollout_invalid_rate', float('nan')):>11.4f}"
        )

    print("─" * 80)
    print("  * = training grid size")


# Experiment 2 - Snake Length
def snake_length_distribution_shift(model_name="unet", n_samples=2000, grid_size=GRID_SIZE, seed=42):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Snake Length Distribution Shift")
    print(f"  Model:      {model_name.upper()}")
    print(f"  Grid size:  {grid_size}x{grid_size}")
    print(f"  Samples:    {n_samples} per bin")
    print(f"  Bins:       {LENGTH_BINS}")
    print("=" * 60)

    random.seed(seed)
    model = load_model(model_name, DEVICE)
    model.eval()

    all_results = {}

    for bin_name, (min_len, max_len) in LENGTH_BINS.items():
        print(f"\n{'─' * 50}")
        print(f"  Bin: {bin_name}  (length {min_len}–{max_len})")
        print(f"{'─' * 50}")

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
                state_np, _, _ = generate_random_snake_state(
                    grid_size=grid_size,
                    min_length=min_len,
                    max_length=max_len,
                )
            except RuntimeError:
                skipped += 1
                continue

            action = random.choice(ACTIONS)
            next_state_np, done = rules_based_next_state(
                state_np, action, grid_size
            )

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
            print(f"  (skipped {skipped} / {n_samples} samples — "
                  f"random walk could not reach length {min_len}–{max_len})")

        total = len(acc["head"])
        if total == 0:
            print("  No samples evaluated for this bin.")
            all_results[bin_name] = {}
            continue

        metrics = compute_metrics(
            acc["head"], acc["done"],
            acc["body_tp"], acc["body_fp"], acc["body_fn"],
            acc["fc_tp"], acc["fc_fp"], acc["fc_fn"],
            acc["food_static_correct"], acc["food_static_total"],
            acc["food_respawn_correct"], acc["food_respawn_total"],
            acc["illegal"], total,
        )
        metrics["n_evaluated"] = total
        print_metrics(metrics, f"Length bin: {bin_name} ({min_len}–{max_len}), "
                               f"n={total}")
        all_results[bin_name] = metrics

    _print_length_shift_summary(all_results)
    return all_results


def _print_length_shift_summary(all_results):
    print("\n" + "=" * 80)
    print("SUMMARY — Snake Length Distribution Shift")
    print("=" * 80)

    header = (f"{'Bin':>10}  {'N':>6}  {'HeadAcc':>8}  {'BodyF1':>7}  "
              f"{'FC_F1':>6}  {'Illegal':>8}  {'FoodStatic':>11}")
    print(header)
    print("─" * 80)

    for bin_name in LENGTH_BINS:
        m = all_results.get(bin_name, {})
        if not m:
            print(f"{bin_name:>10}  {'–':>6}")
            continue
        min_l, max_l = LENGTH_BINS[bin_name]
        label = f"{bin_name}"
        print(
            f"{label:>10}  "
            f"{m.get('n_evaluated', 0):>6}  "
            f"{m.get('head_acc', float('nan')):>8.4f}  "
            f"{m.get('body_f1', float('nan')):>7.4f}  "
            f"{m.get('fc_f1', float('nan')):>6.4f}  "
            f"{m.get('illegal_rate', float('nan')):>8.4f}  "
            f"{m.get('food_static_acc', float('nan')):>11.4f}"
        )

    print("─" * 80)


# Add experiments here
def run_all_experiments():
    # grid_size_generalisation()
    snake_length_distribution_shift()


if __name__ == "__main__":
    run_all_experiments()
