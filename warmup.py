import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from networks.actor import ActorNetworkPolicy  
from networks.actor import StateDependentPolicy

# ====== set path and parameters ======
expert_state_path = "experts/StickInsect_states_v1.pt"
expert_action_path = "experts/StickInsect_actions_v1.pt"
model_save_path = "weights/bc_sac_pretrained_actor.pth"

batch_size = 128
learning_rate = 1e-4
epochs = 1000

# ====== load data ======
states = torch.load(expert_state_path, weights_only=True)   # shape: (num_trajectories, traj_length, state_dim)
actions = torch.load(expert_action_path, weights_only=True)  # shape: (num_trajectories, traj_length, action_dim)

states = torch.tensor(states, dtype=torch.float32).clone().detach()
actions = torch.tensor(actions, dtype=torch.float32).clone().detach()

# ====== build data loader ======
dataset = TensorDataset(states, actions)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ====== initialize Actor network ======
state_dim = states.shape[1]
action_dim = actions.shape[1]

# actor = ActorNetworkPolicy(
#     state_shape=(state_dim,),
#     action_shape=(action_dim,),
#     hidden_units=(64, 64),
#     scale=1.0  # ⚠️ consistent with the scale used in the training of the expert policy
# )

actor = StateDependentPolicy(
            state_shape=(state_dim,),
            action_shape=(action_dim,),
            hidden_units=(256, 256),
            hidden_activation=nn.ReLU(inplace=True)
        )

actor.train()
optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# ====== start training ======
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_states, batch_actions in dataloader:
        pred_actions = actor(batch_states)
        loss = criterion(pred_actions, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_states.size(0)

    avg_loss = epoch_loss / len(dataloader.dataset)
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

# ====== save model parameters ======
torch.save(actor.state_dict(), model_save_path)
print(f"Pretrained actor saved to {model_save_path}")
