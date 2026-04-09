import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

print("Loading data...")
data = np.load("snake_transitions.npz")

states = data["states"]
actions = data["actions"]
next_states = data["next_states"]
dones = data["dones"]  # ✅ added

# NHWC → NCHW
states = np.transpose(states, (0, 3, 1, 2))
next_states = np.transpose(next_states, (0, 3, 1, 2))


# -------------------------
# Dataset
# -------------------------
class SnakeDataset(Dataset):
    def __init__(self, states, actions, next_states, dones):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.dones[idx],
        )


# Train / Val split
train_states, val_states, train_actions, val_actions, train_next, val_next, train_done, val_done = train_test_split(
    states, actions, next_states, dones, test_size=0.2, random_state=42
)

train_dataset = SnakeDataset(train_states, train_actions, train_next, train_done)
val_dataset = SnakeDataset(val_states, val_actions, val_next, val_done)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# -------------------------
# Model
# -------------------------
class BaselineCNNModel(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()

        self.action_embed = nn.Embedding(action_dim, 10 * 10)

        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        # Separate heads (IMPORTANT)
        self.head_out = nn.Conv2d(32, 1, 1)
        self.body_out = nn.Conv2d(32, 1, 1)
        self.food_out = nn.Conv2d(32, 1, 1)

        # Done head
        self.done_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        B = state.size(0)

        action_map = self.action_embed(action).view(B, 1, 10, 10)

        x = torch.cat([state, action_map], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten outputs to (B,100)
        head_logits = self.head_out(x).view(B, -1)
        body_logits = self.body_out(x).view(B, -1)
        food_logits = self.food_out(x).view(B, -1)

        done_logit = self.done_head(x).squeeze(-1)

        return head_logits, body_logits, food_logits, done_logit


# -------------------------
# Training
# -------------------------
if __name__ == "__main__":
    model = BaselineCNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")

    best_loss = float("inf")

    for epoch in range(50):

        model.train()
        total_train_loss = 0

        for state, action, next_state, done in train_loader:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            done = done.to(device)

            head_logits, body_logits, food_logits, done_logit = model(state, action)

            B, _, H, W = state.shape

            # -------- Targets --------
            next_body = next_state[:, 0].view(B, -1)
            next_head = next_state[:, 1].view(B, -1)
            next_food = next_state[:, 2].view(B, -1)

            head_target = next_head.argmax(dim=1)
            food_target = next_food.argmax(dim=1)

            # -------- Losses --------
            loss_head = F.cross_entropy(head_logits, head_target)
            loss_food = F.cross_entropy(food_logits, food_target)

            loss_body = F.binary_cross_entropy_with_logits(body_logits, next_body)

            loss_done = F.binary_cross_entropy_with_logits(done_logit, done)

            loss = loss_head + loss_food + loss_body + loss_done

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            torch.save(model.state_dict(), "baseline_cnn_world_model.pt")
            print("Loss improved, model saved.")

    print("Training complete. Model saved.")
