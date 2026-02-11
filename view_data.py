import numpy as np
import matplotlib.pyplot as plt

data = np.load('snake_transitions.npz')

print("Keys in file:", data.files)

states = data['states']
actions = data['actions']
next_states = data['next_states']
dones = data['dones']

print("\nShapes:")
print("states:", states.shape)
print("actions:", actions.shape)
print("next_states:", next_states.shape)
print("dones:", dones.shape)


def show_state(state, title="State"):
    # Collapse channels into single display grid for viewing
    # body = green, head = brighter green, food = red

    display = np.zeros((state.shape[0], state.shape[1], 3))

    body = state[:, :, 0]
    head = state[:, :, 1]
    food = state[:, :, 2]

    display[:, :, 1] = body * 0.6  # green
    display[:, :, 1] += head * 1.0  # brighter green
    display[:, :, 0] = food * 1.0  # red

    plt.imshow(display)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Show one example
show_state(states[0], "State")
show_state(next_states[0], "Next State")
