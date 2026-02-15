import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

data = np.load("snake_transitions.npz")

states = data["states"]
actions = data["actions"]
next_states = data["next_states"]

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


small_n = 512
dataset = SnakeDataset(
    states[:small_n],
    actions[:small_n],
    next_states[:small_n]
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)


class WorldModel(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()

        self.action_embed = nn.Embedding(action_dim, 10 * 10)

        self.conv1 = nn.Conv2d(3 + 1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, state, action):
        batch_size = state.size(0)

        # Embed action and reshape to spatial map
        action_map = self.action_embed(action)
        action_map = action_map.view(batch_size, 1, 10, 10)

        x = torch.cat([state, action_map], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))

        return x


model = WorldModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(50):
    total_loss = 0

    for state, action, next_state in loader:
        pred = model(state, action)
        loss = criterion(pred, next_state)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}")
