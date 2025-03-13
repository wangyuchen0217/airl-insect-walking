import torch
import numpy as np
import gymnasium as gym


def load_expert_data(state_file, action_file, save_npz=False, npz_filename="expert_data.npz"):
    # Load data from .pt files.
    states = torch.load(state_file, weights_only=True)   # shape: (num_trajectories, traj_length, state_dim)
    actions = torch.load(action_file, weights_only=True)   # shape: (num_trajectories, traj_length, action_dim)
    
    # Convert to numpy arrays.
    states_np = states.numpy()[:,:,:27]
    actions_np = actions.numpy()
    print(f"Load data states: {states_np.shape}, actions: {actions_np.shape}")

    num_trajectories, traj_length, _ = states_np.shape

    states = []
    actions = []
    next_states = []
    for i in range(num_trajectories):
        states.append(states_np[i, :-1, :]) # remove the last state
        actions.append(actions_np[i, :-1, :]) # remove the last action
        next_states.append(states_np[i, 1:, :]) # remove the first state
    states = np.concatenate(states, axis=0)      
    actions = np.concatenate(actions, axis=0)      
    next_states = np.concatenate(next_states, axis=0)
    dones = np.zeros(((traj_length-1)*num_trajectories), dtype=np.float32)
    rewards = np.zeros(((traj_length-1)*num_trajectories), dtype=np.float32)

    expert_data = {
        'state': states,
        'action': actions,
        'reward': rewards,
        'done': dones,
        'next_state': next_states
    }
    print(f"Expert data states: {expert_data['state'].shape}, actions: {expert_data['action'].shape}")

    if save_npz:
        np.savez(npz_filename, **expert_data)

    return expert_data


class ExpertBuffer:
    def __init__(self, expert_data, device):
        self.states = torch.tensor(expert_data['state'], dtype=torch.float32, device=device)
        self.actions = torch.tensor(expert_data['action'], dtype=torch.float32, device=device)
        self.rewards = torch.tensor(expert_data['reward'], dtype=torch.float32, device=device)
        self.dones = torch.tensor(expert_data['done'], dtype=torch.float32, device=device)
        self.next_states = torch.tensor(expert_data['next_state'], dtype=torch.float32, device=device)
        self.size = self.states.size(0)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.dones[idx],
                self.next_states[idx])


if __name__ == "__main__":
    STATE_FILE = "Ant_states.pt"
    ACTION_FILE = "Ant_actions.pt"
    expert_data = load_expert_data(STATE_FILE, ACTION_FILE, save_npz=False, npz_filename="expert_data.npz")
