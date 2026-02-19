import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


# Downsample Block
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res(x)
        return x


# Upsample Block
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # FIX: match spatial size exactly
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = x + skip
        x = self.res(x)
        return x


# FiLM Conditioning
class FiLM(nn.Module):
    def __init__(self, num_actions, channels):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, channels * 2)

    def forward(self, x, action):
        # x shape: (B, C, H, W)
        gamma_beta = self.embedding(action)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta


# Full Model
class ResidualUNetWorldModel(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()

        # Initial
        self.input_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.res1 = ResBlock(32)

        # Encoder
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)

        # FiLM at bottleneck
        self.film = FiLM(num_actions, 128)

        # Decoder
        self.up1 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)

        # Output heads
        self.head_out = nn.Conv2d(32, 1, 1)  # logits for head
        self.body_out = nn.Conv2d(32, 1, 1)  # logits for body
        self.food_out = nn.Conv2d(32, 1, 1)  # logits for food

    def forward(self, state, action):
        # state: (B,3,H,W)
        # action: (B)

        x1 = F.relu(self.input_conv(state))
        x1 = self.res1(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # FiLM conditioning
        x3 = self.film(x3, action)

        # Decode with skip connections
        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        # Output heads (logits)
        head_logits = self.head_out(x)
        body_logits = self.body_out(x)
        food_logits = self.food_out(x)

        return head_logits, body_logits, food_logits


if __name__ == "__main__":
    # Train
    print("Starting training...")

    # Config
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DATA_PATH = "snake_transitions.npz"

    print("Using device ", DEVICE)


    class SnakeDataset(Dataset):
        def __init__(self, path):
            data = np.load(path)

            self.states = data["states"]  # (N,H,W,3)
            self.actions = data["actions"]  # (N,)
            self.next_states = data["next_states"]  # (N,H,W,3)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            state = torch.tensor(self.states[idx], dtype=torch.float32)
            next_state = torch.tensor(self.next_states[idx], dtype=torch.float32)
            action = torch.tensor(self.actions[idx], dtype=torch.long)

            # Convert to (C,H,W)
            state = state.permute(2, 0, 1)
            next_state = next_state.permute(2, 0, 1)

            return state, action, next_state


    dataset = SnakeDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ResidualUNetWorldModel(num_actions=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        total_loss = 0

        for states, actions, next_states in loader:
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            next_states = next_states.to(DEVICE)

            head_logits, body_logits, food_logits = model(states, actions)

            # ----------------------------------
            # Targets
            # ----------------------------------

            # Body target
            target_body = next_states[:, 0:1]  # channel 0

            # Head target -> convert one-hot grid to index
            target_head_map = next_states[:, 1]  # (B,H,W)
            B, H, W = target_head_map.shape
            target_head_index = target_head_map.view(B, -1).argmax(dim=1)

            # Food target
            target_food = next_states[:, 2:3]

            head_logits_flat = head_logits.view(B, -1)
            loss_head = F.cross_entropy(head_logits_flat, target_head_index)

            # Body
            loss_body = F.binary_cross_entropy_with_logits(body_logits, target_body)

            # Food
            loss_food = F.binary_cross_entropy_with_logits(food_logits, target_food)

            loss = loss_head + loss_body + loss_food

            # ----------------------------------
            # Backprop
            # ----------------------------------

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

        torch.save(model.state_dict(), "residual_unet_world_model.pt")
        print("Model saved.")