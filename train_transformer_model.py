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


# Dataset
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
        state = self.states[idx]
        action = self.actions[idx]
        next_state = self.next_states[idx]
        done = self.dones[idx]

        state = torch.tensor(state).permute(2, 0, 1).float()
        next_state = torch.tensor(next_state).permute(2, 0, 1).float()
        action = torch.tensor(action).long()
        done = torch.tensor(done).float()

        return state, action, next_state, done


# Model
class TransformerWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cell_embed = nn.Linear(STATE_CHANNELS, EMBED_DIM)
        self.action_embed = nn.Embedding(4, EMBED_DIM)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, NUM_CELLS + 1, EMBED_DIM)
        )

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

        self.head_out = nn.Linear(EMBED_DIM, 1)
        self.body_out = nn.Linear(EMBED_DIM, 1)
        self.food_out = nn.Linear(EMBED_DIM, 1)
        self.done_out = nn.Linear(EMBED_DIM, 1)

    def forward(self, state, action):
        B = state.size(0)

        state = state.reshape(B, STATE_CHANNELS, -1)
        state = state.permute(0, 2, 1)

        cell_tokens = self.cell_embed(state)

        action_token = self.action_embed(action).unsqueeze(1)

        tokens = torch.cat([action_token, cell_tokens], dim=1)
        tokens = tokens + self.pos_embedding

        tokens = self.transformer(tokens)

        action_context = tokens[:, 0]
        grid_tokens = tokens[:, 1:]

        head_logits = self.head_out(grid_tokens).squeeze(-1)
        body_logits = self.body_out(grid_tokens).squeeze(-1)
        food_logits = self.food_out(grid_tokens).squeeze(-1)

        done_logit = self.done_out(action_context).squeeze(-1)

        return head_logits, body_logits, food_logits, done_logit


# Helper: find tail cell
def find_tail_mask(body, head):
    """
    body: (B,100)
    head: (B,100)
    returns tail mask (B,100)
    """
    B = body.size(0)

    body_grid = body.reshape(B, GRID_SIZE, GRID_SIZE)
    head_grid = head.reshape(B, GRID_SIZE, GRID_SIZE)

    padded = F.pad(body_grid, (1,1,1,1))

    neighbours = (
        padded[:, :-2, 1:-1] +
        padded[:, 2:, 1:-1] +
        padded[:, 1:-1, :-2] +
        padded[:, 1:-1, 2:]
    )

    tail = (body_grid == 1) & (neighbours == 1) & (head_grid == 0)

    return tail.reshape(B, NUM_CELLS).float()


# Training loop
def train():
    dataset = SnakeDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerWorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_weight = torch.tensor(8.0, device=DEVICE)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for state, action, next_state, done in loader:

            state = state.to(DEVICE)
            action = action.to(DEVICE)
            next_state = next_state.to(DEVICE)
            done = done.to(DEVICE)

            head_logits, body_logits, food_logits, done_logit = model(state, action)

            next_body = next_state[:,0].reshape(-1, NUM_CELLS)
            next_head = next_state[:,1].reshape(-1, NUM_CELLS)
            next_food = next_state[:,2].reshape(-1, NUM_CELLS)

            head_target = torch.argmax(next_head, dim=1)
            food_target = torch.argmax(next_food, dim=1)

            # Standard losses
            head_loss = F.cross_entropy(head_logits, head_target)
            food_loss = F.cross_entropy(food_logits, food_target)

            body_loss = F.binary_cross_entropy_with_logits(
                body_logits,
                next_body,
                pos_weight=pos_weight
            )

            done_loss = F.binary_cross_entropy_with_logits(done_logit, done)

            # Body size regularisation
            pred_body_prob = torch.sigmoid(body_logits)
            pred_body_sum = pred_body_prob.sum(dim=1)
            true_body_sum = next_body.sum(dim=1)

            size_loss = F.mse_loss(pred_body_sum, true_body_sum)

            # ---------- Tail dynamics loss ----------

            prev_body = state[:,0].reshape(-1, NUM_CELLS)
            prev_head = state[:,1].reshape(-1, NUM_CELLS)

            tail_mask = find_tail_mask(prev_body, prev_head).to(DEVICE)

            tail_prob = (pred_body_prob * tail_mask).sum(dim=1)

            food_eaten = (true_body_sum > prev_body.sum(dim=1)).float()

            tail_loss = ((1 - food_eaten) * tail_prob).mean()

            # ---------- Final loss ----------

            loss = (
                head_loss
                + food_loss
                + 2.0 * body_loss
                + done_loss
                + 0.1 * size_loss
                + 0.5 * tail_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model improved - weights saved.")

    print("Training complete.")


if __name__ == "__main__":
    train()