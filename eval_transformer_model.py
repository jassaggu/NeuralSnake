import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import random
from train_transformer_model import TransformerWorldModel
from eval_tools import get_next_logical_frame

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "snake_transitions.npz"
MODEL_PATH = "transformer_world_model.pt"
GRID_SIZE = 10
BATCH_SIZE = 128
NUM_CELLS = GRID_SIZE * GRID_SIZE


class SnakeDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.states = data["states"]  # (N, H, W, C)
        self.actions = data["actions"]  # (N,)
        self.next_states = data["next_states"]
        self.dones = data["dones"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        next_state = self.next_states[idx]
        done = self.dones[idx]

        # Convert (H,W,C) -> (C,H,W)
        state = torch.tensor(state).permute(2, 0, 1).float()
        next_state = torch.tensor(next_state).permute(2, 0, 1).float()
        action = torch.tensor(action).long()
        done = torch.tensor(done).float()

        return state, action, next_state, done


def evaluate(model, loader):
    model.eval()

    total = 0

    head_correct = 0
    food_correct = 0
    done_correct = 0

    body_tp = 0
    body_fp = 0
    body_fn = 0

    food_consumption_correct = 0
    illegal_states = 0  # illegal states determination is incomplete, doesnt incldue body joinedness etc

    with torch.no_grad():
        for state, action, next_state, done in loader:
            state = state.to(DEVICE)
            action = action.to(DEVICE)
            next_state = next_state.to(DEVICE)
            done = done.to(DEVICE)

            head_logits, body_logits, food_logits, done_logit = model(state, action)

            B = state.size(0)
            total += B

            # ---------- Targets ----------
            next_body = next_state[:, 0].view(B, NUM_CELLS)
            next_head = next_state[:, 1].view(B, NUM_CELLS)
            next_food = next_state[:, 2].view(B, NUM_CELLS)

            head_target = torch.argmax(next_head, dim=1)
            food_target = torch.argmax(next_food, dim=1)

            # Food consumption ground truth:
            # food disappears if head moves into food
            prev_head = state[:, 1].view(B, NUM_CELLS)
            prev_food = state[:, 2].view(B, NUM_CELLS)

            prev_head_idx = torch.argmax(prev_head, dim=1)
            prev_food_idx = torch.argmax(prev_food, dim=1)

            true_food_consumed = (prev_head_idx == prev_food_idx)

            # ---------- Predictions ----------
            head_pred = torch.argmax(head_logits, dim=1)
            food_pred = torch.argmax(food_logits, dim=1)
            done_pred = (torch.sigmoid(done_logit) > 0.5).float()
            body_pred = (torch.sigmoid(body_logits) > 0.5).float()

            # ---------- Accuracy Metrics ----------
            head_correct += (head_pred == head_target).sum().item()
            food_correct += (food_pred == food_target).sum().item()
            done_correct += (done_pred == done).sum().item()

            # ---------- Body F1 ----------
            body_tp += ((body_pred.view(B, -1) == 1) & (next_body == 1)).sum().item()
            body_fp += ((body_pred.view(B, -1) == 1) & (next_body == 0)).sum().item()
            body_fn += ((body_pred.view(B, -1) == 0) & (next_body == 1)).sum().item()

            # ---------- Food Consumption Accuracy ----------
            pred_food_consumed = (head_pred == prev_food_idx)
            food_consumption_correct += (pred_food_consumed == true_food_consumed).sum().item()

            # ---------- Illegal State Check ----------
            for i in range(B):

                head_map = torch.zeros(NUM_CELLS)
                head_map[head_pred[i]] = 1
                head_map = head_map.view(GRID_SIZE, GRID_SIZE)

                food_map = torch.zeros(NUM_CELLS)
                food_map[food_pred[i]] = 1
                food_map = food_map.view(GRID_SIZE, GRID_SIZE)

                # FIX: reshape full body prediction
                body_map = body_pred[i].view(GRID_SIZE, GRID_SIZE).cpu()

                state_tensor = torch.stack([body_map, head_map, food_map])

                if not is_valid_state(state_tensor):
                    illegal_states += 1

    head_acc = head_correct / total
    food_acc = food_correct / total
    done_acc = done_correct / total

    precision = body_tp / (body_tp + body_fp + 1e-8)
    recall = body_tp / (body_tp + body_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    food_consumption_acc = food_consumption_correct / total
    illegal_rate = illegal_states / total

    print("\n=== Single-Step Evaluation ===")
    print(f"Head Accuracy: {head_acc:.4f}")
    print(f"Food Position Accuracy: {food_acc:.4f}")
    print(f"Done Accuracy: {done_acc:.4f}")
    print(f"Body F1 Score: {f1:.4f}")
    print(f"Food Consumption Accuracy: {food_consumption_acc:.4f}")
    print(f"Illegal State Rate: {illegal_rate:.4f}")


def is_valid_state(state):
    body = state[0]
    head = state[1]
    food = state[2]

    if head.sum() != 1:
        return False

    if food.sum() != 1:
        return False

    # head overlapping body
    if (body * head).sum() > 0:
        return False

    return True


def rollout_evaluation(model, test_dataset, num_trials=100, max_steps=50):
    model.eval()

    survival_lengths = []
    illegal_count = 0

    for _ in range(num_trials):

        idx = random.randint(0, len(test_dataset) - 1)
        state, action, next_state, done = test_dataset[idx]

        # Convert initial state to numpy (H,W,C)
        current_state = state.permute(1, 2, 0).cpu().numpy()

        steps_survived = 0

        for _ in range(max_steps):

            action = random.randint(0, 3)

            # ---------- Model prediction ----------
            state_tensor = torch.tensor(current_state).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            action_tensor = torch.tensor([action]).to(DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, done_logit = model(state_tensor, action_tensor)

            head_idx = torch.argmax(head_logits, dim=1).item()
            food_idx = torch.argmax(food_logits, dim=1).item()
            body_map = (torch.sigmoid(body_logits) > 0.5).float().view(GRID_SIZE, GRID_SIZE).cpu().numpy()

            # Build predicted state (H, W, C)
            pred_state = np.zeros((GRID_SIZE, GRID_SIZE, 3))

            # Body
            pred_state[:, :, 0] = body_map

            # Head
            hx, hy = head_idx % GRID_SIZE, head_idx // GRID_SIZE
            pred_state[hy, hx, 1] = 1

            # Food
            fx, fy = food_idx % GRID_SIZE, food_idx // GRID_SIZE
            pred_state[fy, fx, 2] = 1

            # ---------- Ground truth logical step ----------
            true_next_state, done_flag = get_next_logical_frame(current_state, action)

            # If logical simulation fails → current state already invalid
            if true_next_state is None:
                illegal_count += 1
                break

            # ---------- Divergence check ----------
            # Compare predicted vs true next state
            if not np.array_equal(pred_state, true_next_state):
                illegal_count += 1
                break

            # Move forward using predicted state (autoregressive rollout)
            current_state = pred_state
            steps_survived += 1

        survival_lengths.append(steps_survived)

    avg_survival = sum(survival_lengths) / len(survival_lengths)
    illegal_rate = illegal_count / num_trials

    print("\n=== Rollout Evaluation ===")
    print(f"Average Rollout Length Before Divergence: {avg_survival:.2f}")
    print(f"Illegal State Rate (Rollout): {illegal_rate:.4f}")


def multi_step_error_curve(model, test_dataset, num_trials=100, max_steps=20):
    model.eval()

    errors_per_step = [[] for _ in range(max_steps)]

    for _ in range(num_trials):

        idx = random.randint(0, len(test_dataset) - 1)
        state, _, _, _ = test_dataset[idx]

        # Start from real state
        current_state = state.permute(1, 2, 0).cpu().numpy()

        for step in range(max_steps):

            action = random.randint(0, 3)

            # ---------- Model prediction ----------
            state_tensor = torch.tensor(current_state).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            action_tensor = torch.tensor([action]).to(DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, _ = model(state_tensor, action_tensor)

            head_idx = torch.argmax(head_logits, dim=1).item()
            body_map = (torch.sigmoid(body_logits) > 0.5).float().view(GRID_SIZE, GRID_SIZE).cpu().numpy()

            pred_state = np.zeros((GRID_SIZE, GRID_SIZE, 3))
            pred_state[:, :, 0] = body_map

            hx, hy = head_idx % GRID_SIZE, head_idx // GRID_SIZE
            pred_state[hy, hx, 1] = 1

            # NOTE: ignore food channel completely

            # ---------- Ground truth ----------
            true_next_state, done_flag = get_next_logical_frame(current_state, action)

            if true_next_state is None:
                break

            # ---------- Compute error (ignore food channel) ----------
            pred_no_food = pred_state[:, :, :2]
            true_no_food = true_next_state[:, :, :2]

            mse = np.mean((pred_no_food - true_no_food) ** 2)
            errors_per_step[step].append(mse)

            # Move forward using TRUE state (not predicted)
            current_state = true_next_state

            if done_flag:
                break

    # ---------- Aggregate ----------
    avg_errors = []
    for step_errors in errors_per_step:
        if len(step_errors) > 0:
            avg_errors.append(np.mean(step_errors))
        else:
            avg_errors.append(0)

    print("\n=== Multi-Step Error Curve ===")
    for i, e in enumerate(avg_errors):
        print(f"Step {i + 1}: MSE = {e:.6f}")

    return avg_errors


model = TransformerWorldModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
dataset = SnakeDataset(DATA_PATH)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

evaluate(model, test_loader)
rollout_evaluation(model, test_dataset)
multi_step_error_curve(model, test_dataset)

"""
=== Single-Step Evaluation ===
Head Accuracy: 0.9998
Food Position Accuracy: 0.8744
Done Accuracy: 0.9998
Body F1 Score: 0.9660
Food Consumption Accuracy: 0.8730
Illegal State Rate: 0.0060

=== Rollout Evaluation ===
Average Rollout Length Before Divergence: 29.80
Illegal State Rate (Rollout): 0.6100
"""
