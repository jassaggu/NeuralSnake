import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
from train_transformer_model import TransformerWorldModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "snake_transitions.npz"
MODEL_PATH = "transformer_world_model.pt"
GRID_SIZE = 10
BATCH_SIZE = 128
NUM_CELLS = GRID_SIZE * GRID_SIZE


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


def evaluate_meta_metrics(model, loader):
    model.eval()

    total_mse = 0
    total = 0

    with torch.no_grad():
        for state, action, next_state, done in loader:
            state = state.to(DEVICE)
            action = action.to(DEVICE)
            next_state = next_state.to(DEVICE)

            head_logits, body_logits, food_logits, done_logit = model(state, action)

            B = state.size(0)

            body_pred = torch.sigmoid(body_logits).view(B, 1, GRID_SIZE, GRID_SIZE)

            head_idx = torch.argmax(head_logits, dim=1)
            head_map = torch.zeros((B, NUM_CELLS), device=DEVICE)
            head_map[torch.arange(B), head_idx] = 1
            head_map = head_map.view(B, 1, GRID_SIZE, GRID_SIZE)

            food_idx = torch.argmax(food_logits, dim=1)
            food_map = torch.zeros((B, NUM_CELLS), device=DEVICE)
            food_map[torch.arange(B), food_idx] = 1
            food_map = food_map.view(B, 1, GRID_SIZE, GRID_SIZE)

            pred_state = torch.cat([body_pred, head_map, food_map], dim=1)

            mse = torch.mean((pred_state - next_state) ** 2).item()

            total_mse += mse * B
            total += B

    state_mse = total_mse / total

    print("\n=== Meta Model Metrics ===")
    print(f"State Prediction Error (MSE): {state_mse:.6f}")


def multi_step_prediction_error(model, dataset, steps=10, trials=200):
    model.eval()

    total_error = 0
    count = 0

    for _ in range(trials):
        idx = random.randint(0, len(dataset) - steps - 1)

        state, action, next_state, done = dataset[idx]
        current_state = state.unsqueeze(0).to(DEVICE)

        for t in range(steps):
            action = torch.tensor([dataset[idx + t][1]], device=DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, done_logit = model(current_state, action)

            body_pred = torch.sigmoid(body_logits).view(1, 1, GRID_SIZE, GRID_SIZE)

            head_idx = torch.argmax(head_logits, dim=1)
            head_map = torch.zeros((1, NUM_CELLS), device=DEVICE)
            head_map[0, head_idx] = 1
            head_map = head_map.view(1, 1, GRID_SIZE, GRID_SIZE)

            food_idx = torch.argmax(food_logits, dim=1)
            food_map = torch.zeros((1, NUM_CELLS), device=DEVICE)
            food_map[0, food_idx] = 1
            food_map = food_map.view(1, 1, GRID_SIZE, GRID_SIZE)

            pred_state = torch.cat([body_pred, head_map, food_map], dim=1)

            true_state = dataset[idx + t][2].unsqueeze(0).to(DEVICE)

            error = torch.mean((pred_state - true_state) ** 2).item()

            total_error += error
            count += 1

            current_state = pred_state.detach()

    mse = total_error / count

    print(f"Multi-step Prediction Error: {mse:.6f}")


def divergence_time(model, dataset, threshold=0.05, max_steps=50, trials=200):
    model.eval()

    divergence_times = []

    for _ in range(trials):

        idx = random.randint(0, len(dataset) - max_steps - 1)
        state, action, next_state, done = dataset[idx]

        current_state = state.unsqueeze(0).to(DEVICE)

        for t in range(max_steps):

            action = torch.tensor([dataset[idx + t][1]], device=DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, done_logit = model(current_state, action)

            body_pred = torch.sigmoid(body_logits).view(1, 1, GRID_SIZE, GRID_SIZE)

            head_idx = torch.argmax(head_logits, dim=1)
            head_map = torch.zeros((1, NUM_CELLS), device=DEVICE)
            head_map[0, head_idx] = 1
            head_map = head_map.view(1, 1, GRID_SIZE, GRID_SIZE)

            food_idx = torch.argmax(food_logits, dim=1)
            food_map = torch.zeros((1, NUM_CELLS), device=DEVICE)
            food_map[0, food_idx] = 1
            food_map = food_map.view(1, 1, GRID_SIZE, GRID_SIZE)

            pred_state = torch.cat([body_pred, head_map, food_map], dim=1)

            true_state = dataset[idx + t][2].unsqueeze(0).to(DEVICE)

            error = torch.mean((pred_state - true_state) ** 2).item()

            if error > threshold:
                divergence_times.append(t)
                break

            current_state = pred_state.detach()

        else:
            divergence_times.append(max_steps)

    avg_divergence = sum(divergence_times) / len(divergence_times)

    print(f"Divergence Time: {avg_divergence:.2f} steps")


def rollout_reward_and_length(model, dataset, trials=100, max_steps=50):
    model.eval()

    rewards = []
    lengths = []

    for _ in range(trials):

        idx = random.randint(0, len(dataset) - 1)
        state, action, next_state, done = dataset[idx]

        current_state = state.unsqueeze(0).to(DEVICE)

        reward = 0
        steps = 0

        for _ in range(max_steps):

            action = torch.randint(0, 4, (1,), device=DEVICE)

            with torch.no_grad():
                head_logits, body_logits, food_logits, done_logit = model(current_state, action)

            done_pred = torch.sigmoid(done_logit).item() > 0.5

            reward += 1
            steps += 1

            if done_pred:
                break

            body_pred = torch.sigmoid(body_logits).view(1, 1, GRID_SIZE, GRID_SIZE)

            head_idx = torch.argmax(head_logits, dim=1)
            head_map = torch.zeros((1, NUM_CELLS), device=DEVICE)
            head_map[0, head_idx] = 1
            head_map = head_map.view(1, 1, GRID_SIZE, GRID_SIZE)

            food_idx = torch.argmax(food_logits, dim=1)
            food_map = torch.zeros((1, NUM_CELLS), device=DEVICE)
            food_map[0, food_idx] = 1
            food_map = food_map.view(1, 1, GRID_SIZE, GRID_SIZE)

            current_state = torch.cat([body_pred, head_map, food_map], dim=1).detach()

        rewards.append(reward)

    avg_reward = sum(rewards) / len(rewards)

    print(f"Average Episode Length/Survival Reward: {avg_reward:.2f}")


model = TransformerWorldModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = SnakeDataset(DATA_PATH)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

evaluate_meta_metrics(model, test_loader)
multi_step_prediction_error(model, test_dataset)
divergence_time(model, test_dataset)
rollout_reward_and_length(model, test_dataset)

"""
=== Meta Model Metrics ===
State Prediction Error (MSE): 0.002258
Multi-step Prediction Error: 0.068679
Divergence Time: 1.20 steps
Average Episode Length/Survival Reward: 31.89
"""
