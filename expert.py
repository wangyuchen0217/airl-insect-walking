import torch
import numpy as np
import gymnasium as gym


def load_expert_data(state_file, action_file, save_npz=False, npz_filename="expert_data.npz"):
    """
    Loads expert data from state and action .pt files and constructs a dictionary。
    Returns:
        dict: Expert data dictionary.
    """
    # Load data from .pt files.
    states = torch.load(state_file, weights_only=True)   # shape: (num_trajectories, traj_length, state_dim)
    actions = torch.load(action_file, weights_only=True)   # shape: (num_trajectories, traj_length, action_dim)
    
    # Convert to numpy arrays.
    states_np = states.numpy()[:,:,:27]
    actions_np = actions.numpy()
    print(f"states: {states_np.shape}, actions: {actions_np.shape}")
    
    num_trajectories, traj_length, state_dim = states_np.shape
    _, _, action_dim = actions_np.shape
    
    # Flatten trajectories: (num_trajectories * traj_length, dim)
    states_flat = states_np.reshape(-1, state_dim)
    actions_flat = actions_np.reshape(-1, action_dim)
    
    # Create dummy rewards as zeros.
    rewards_flat = np.zeros((num_trajectories * traj_length,), dtype=np.float32)
    
    # Create done flags: set last timestep of each trajectory to 1, others 0.
    dones = np.zeros((num_trajectories, traj_length), dtype=np.float32)
    dones[:, -1] = 1.0
    dones_flat = dones.reshape(-1, 1)
    
    # Create next_states: shift states along the time dimension;
    # for the last timestep of each trajectory, simply copy the last state.
    next_states = np.zeros_like(states_np)
    next_states[:, :-1, :] = states_np[:, 1:, :]
    next_states[:, -1, :] = states_np[:, -1, :]
    next_states_flat = next_states.reshape(-1, state_dim)
    
    expert_data = {
        'state': states_flat,
        'action': actions_flat,
        'reward': rewards_flat,
        'done': dones_flat,
        'next_state': next_states_flat
    }
    
    if save_npz:
        np.savez(npz_filename, **expert_data)
    
    return expert_data

class ExpertBuffer:
    def __init__(self, expert_data, device):
        """
        Initializes the ExpertBuffer by converting the expert demonstration data 
        into PyTorch tensors and moving them to the specified device.
        """
        self.states = torch.tensor(expert_data['state'], dtype=torch.float32, device=device)
        self.actions = torch.tensor(expert_data['action'], dtype=torch.float32, device=device)
        self.rewards = torch.tensor(expert_data['reward'], dtype=torch.float32, device=device)
        self.dones = torch.tensor(expert_data['done'], dtype=torch.float32, device=device)
        self.next_states = torch.tensor(expert_data['next_state'], dtype=torch.float32, device=device)
        self.size = self.states.size(0)

    def sample(self, batch_size):
        """
        Randomly samples a mini-batch of expert data.
        """
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
