import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# -------------------------
# Residual Block (GroupNorm)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


# -------------------------
# Downsample
# -------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.gn = nn.GroupNorm(8, out_ch)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        x = F.relu(self.gn(self.conv(x)))
        return self.res(x)


# -------------------------
# Upsample
# -------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = x + skip
        return self.res(x)


# -------------------------
# FiLM
# -------------------------
class FiLM(nn.Module):
    def __init__(self, num_actions, channels):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, channels * 2)

    def forward(self, x, action):
        gamma_beta = self.embedding(action)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# -------------------------
# World Model
# -------------------------
class ResidualUNetWorldModel(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()

        self.input_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.res1 = ResBlock(32)

        self.down1 = DownBlock(32, 64)

        self.film = FiLM(num_actions, 64)

        self.up1 = UpBlock(64, 32)

        self.head_out = nn.Conv2d(32, 1, 1)
        self.body_out = nn.Conv2d(32, 1, 1)
        self.food_out = nn.Conv2d(32, 1, 1)

        # ✅ ADDED
        self.done_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        x1 = F.relu(self.input_conv(state))
        x1 = self.res1(x1)

        x2 = self.down1(x1)
        x2 = self.film(x2, action)

        x = self.up1(x2, x1)

        # ✅ Flatten to (B, 100)
        head_logits = self.head_out(x).view(x.size(0), -1)
        body_logits = self.body_out(x).view(x.size(0), -1)
        food_logits = self.food_out(x).view(x.size(0), -1)

        # ✅ Added done prediction
        done_logit = self.done_head(x).squeeze(-1)

        return head_logits, body_logits, food_logits, done_logit


# -------------------------
# Dataset
# -------------------------
class SnakeDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.states = data["states"]
        self.actions = data["actions"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]  # ✅ ADDED

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32).permute(2, 0, 1)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float32).permute(2, 0, 1)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        done = torch.tensor(self.dones[idx], dtype=torch.float32)  # ✅ ADDED

        return state, action, next_state, done


# -------------------------
# Training
# -------------------------
if __name__ == "__main__":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    DATA_PATH = "snake_transitions.npz"

    dataset = SnakeDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ResidualUNetWorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_weight_body = torch.tensor([5.0]).to(DEVICE)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for states, actions, next_states, dones in loader:  # ✅ UPDATED
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            next_states = next_states.to(DEVICE)
            dones = dones.to(DEVICE)

            # ✅ UPDATED unpack
            head_logits, body_logits, food_logits, done_logit = model(states, actions)

            B, _, H, W = states.shape

            # -------- Head Loss --------
            target_head = next_states[:, 1].view(B, -1)
            head_target_idx = target_head.argmax(dim=1)
            loss_head = F.cross_entropy(head_logits, head_target_idx)

            # -------- Food Loss --------
            target_food = next_states[:, 2].view(B, -1)
            food_target_idx = target_food.argmax(dim=1)
            loss_food = F.cross_entropy(food_logits, food_target_idx)

            # -------- Body Loss --------
            target_body = next_states[:, 0].view(B, -1)
            loss_body = F.binary_cross_entropy_with_logits(
                body_logits, target_body, pos_weight=pos_weight_body
            )

            # -------- Done Loss --------
            loss_done = F.binary_cross_entropy_with_logits(done_logit, dones)

            # -------- Movement Penalty --------
            prev_head = states[:, 1].view(B, -1)
            prev_head_idx = prev_head.argmax(dim=1)
            pred_head_idx = head_logits.argmax(dim=1)

            prev_x, prev_y = prev_head_idx // W, prev_head_idx % W
            pred_x, pred_y = pred_head_idx // W, pred_head_idx % W

            movement_dist = torch.abs(prev_x - pred_x) + torch.abs(prev_y - pred_y)
            movement_penalty = torch.mean((movement_dist - 1).clamp(min=0).float())

            # -------- Total Loss --------
            loss = (
                2.0 * loss_head
                + 0.5 * loss_body
                + 1.5 * loss_food
                + 0.5 * movement_penalty
                + 1.0 * loss_done
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

        torch.save(model.state_dict(), "unet_world_model.pt")

    print("Training complete.")