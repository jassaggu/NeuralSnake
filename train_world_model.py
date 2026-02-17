import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Device setup for Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

print("Loading data...")
data = np.load("snake_transitions.npz")

states = data["states"]
actions = data["actions"]
next_states = data["next_states"]

# Convert to NCHW
states = np.transpose(states, (0, 3, 1, 2))
next_states = np.transpose(next_states, (0, 3, 1, 2))


class SnakeDataset(Dataset):
    def __init__(self, states, actions, next_states):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


# Train / Validation split
train_states, val_states, train_actions, val_actions, train_next, val_next = train_test_split(
    states, actions, next_states, test_size=0.2, random_state=42
)

train_dataset = SnakeDataset(train_states, train_actions, train_next)
val_dataset = SnakeDataset(val_states, val_actions, val_next)

# Increased batch size for GPU utilization
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)


class WorldModel(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()

        self.action_embed = nn.Embedding(action_dim, 10 * 10)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, state, action):
        batch_size = state.size(0)

        action_map = self.action_embed(action)
        action_map = action_map.view(batch_size, 1, 10, 10)

        x = torch.cat([state, action_map], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))

        return x


if __name__ == "__main__":
    model = WorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print("Starting training...")

    print("Starting training...")

    for epoch in range(50):

        model.train()
        total_train_loss = 0

        for state, action, next_state in train_loader:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)

            pred = model(state, action)
            loss = criterion(pred, next_state)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_correct = 0
        total_cells = 0
        total_head_error = 0
        total_samples = 0

        with torch.no_grad():
            for state, action, next_state in val_loader:
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)

                pred = model(state, action)
                loss = criterion(pred, next_state)

                total_val_loss += loss.item()

                # ---- Per-cell accuracy ----
                total_correct += (pred.round() == next_state).float().sum().item()
                total_cells += next_state.numel()

                # ---- Head spatial error ----
                pred_head_map = pred[:, 0]
                true_head_map = next_state[:, 0]

                pred_coords = pred_head_map.view(pred.size(0), -1).argmax(dim=1)
                true_coords = true_head_map.view(next_state.size(0), -1).argmax(dim=1)

                pred_x = pred_coords // 10
                pred_y = pred_coords % 10
                true_x = true_coords // 10
                true_y = true_coords % 10

                dist = torch.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

                total_head_error += dist.sum().item()
                total_samples += dist.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        cell_accuracy = total_correct / total_cells
        avg_head_error = total_head_error / total_samples

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Cell Acc: {cell_accuracy:.4f} | "
            f"Head Dist: {avg_head_error:.4f}"
        )

    torch.save(model.state_dict(), "world_model.pt")
    print("Training complete. Model saved.")
