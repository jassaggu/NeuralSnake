import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config
GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE
STATE_CHANNELS = 3
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
BATCH_SIZE = 128
EPOCHS = 25
LR = 3e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "snake_transitions.npz"
MODEL_PATH = "transformer_world_model.pt"


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


class TransformerWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cell_embed = nn.Linear(STATE_CHANNELS, EMBED_DIM)
        self.action_embed = nn.Embedding(4, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        # Output heads
        self.head_out = nn.Linear(EMBED_DIM, 1)
        self.body_out = nn.Linear(EMBED_DIM, 1)
        self.food_out = nn.Linear(EMBED_DIM, 1)

        self.done_out = nn.Linear(EMBED_DIM, 1)

    def forward(self, state, action):
        B = state.size(0)

        # Flatten grid into tokens
        state = state.view(B, STATE_CHANNELS, -1)  # (B,C,100)
        state = state.permute(0, 2, 1)  # (B,100,C)

        cell_tokens = self.cell_embed(state)  # (B,100,E)

        action_token = self.action_embed(action)  # (B,E)
        action_token = action_token.unsqueeze(1)  # (B,1,E)

        tokens = torch.cat([action_token, cell_tokens], dim=1)
        tokens = self.transformer(tokens)  # (B,101,E)

        action_context = tokens[:, 0]  # (B,E)
        grid_tokens = tokens[:, 1:]  # (B,100,E)

        head_logits = self.head_out(grid_tokens).squeeze(-1)
        body_logits = self.body_out(grid_tokens).squeeze(-1)
        food_logits = self.food_out(grid_tokens).squeeze(-1)

        done_logit = self.done_out(action_context).squeeze(-1)

        return head_logits, body_logits, food_logits, done_logit


def train():
    dataset = SnakeDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerWorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        total_loss = 0

        for state, action, next_state, done in loader:
            state = state.to(DEVICE)
            action = action.to(DEVICE)
            next_state = next_state.to(DEVICE)
            done = done.to(DEVICE)

            head_logits, body_logits, food_logits, done_logit = model(state, action)

            # ---- Targets ----

            next_body = next_state[:, 0].view(-1, NUM_CELLS)
            next_head = next_state[:, 1].view(-1, NUM_CELLS)
            next_food = next_state[:, 2].view(-1, NUM_CELLS)

            head_target = torch.argmax(next_head, dim=1)
            food_target = torch.argmax(next_food, dim=1)

            # ---- Losses ----

            head_loss = F.cross_entropy(head_logits, head_target)
            food_loss = F.cross_entropy(food_logits, food_target)
            body_loss = F.binary_cross_entropy_with_logits(body_logits, next_body)
            done_loss = F.binary_cross_entropy_with_logits(done_logit, done)

            loss = head_loss + food_loss + body_loss + done_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")


if __name__ == "__main__":
    train()
