import torch
import numpy as np
import gymnasium as gym

def collect_agent_trajectories(env, policy, steps_per_iter, device):
    """
    Run the current policy to collect a number of transitions.
    (Gymnasium’s reset() returns (obs, info) and step() returns (obs, reward, done, truncated, info)).
    """
    trajectories = []
    state, _ = env.reset()
    for _ in range(steps_per_iter):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _ = policy.act(state_tensor)
        action_np = action.cpu().numpy()[0]
        next_state, reward, done, truncated, info = env.step(action_np)
        trajectories.append({
            'state': state,
            'action': action_np,
            'next_state': next_state,
            'log_prob': log_prob.cpu().item(),
            'done': done or truncated
        })
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    return trajectories

# def load_expert_data(model_name='Ant'):
#     expert_states = torch.load(model_name+'_states.pt').numpy() # (62, 1000, 111)
#     expert_actions = torch.load(model_name+'_actions.pt').numpy() # (62, 1000, 8)
#     print("expert_states:", expert_states.shape, "expert_actions:", expert_actions.shape)
#     # take only 1 trail
#     expert_states = expert_states[:,:,:27].reshape(-1,27) # (62000, 27)
#     expert_actions = expert_actions.reshape(-1,8) # (62000, 8)
#     expert_next_states = np.roll(expert_states, -1, axis=0)
#     expert_next_states[-1] = expert_states[-1]
#     print("expert_states:", expert_states.shape, "expert_actions:", expert_actions.shape, "expert_next_states:", expert_next_states.shape)
#     return expert_states, expert_actions, expert_next_states

def load_expert_data(state_file, action_file, save_npz=False, npz_filename="expert_data.npz"):
    """
    Loads expert data from state and action .pt files and constructs a dictionary containing:
        - 'state': numpy array of shape (num_trajectories * traj_length, state_dim)
        - 'action': numpy array of shape (num_trajectories * traj_length, action_dim)
        - 'reward': numpy array of shape (num_trajectories * traj_length,) with zeros
        - 'done': numpy array of shape (num_trajectories * traj_length, 1), where only the last timestep of each trajectory is 1
        - 'next_state': numpy array of shape (num_trajectories * traj_length, state_dim)
    
    Args:
        state_file (str): Path to the state.pt file. Expected shape: (num_trajectories, traj_length, state_dim)
        action_file (str): Path to the action.pt file. Expected shape: (num_trajectories, traj_length, action_dim)
        save_npz (bool): If True, saves the resulting dictionary as an .npz file.
        npz_filename (str): The filename for the saved npz file.
    
    Returns:
        dict: Expert data dictionary.
    """
    # Load data from .pt files.
    states = torch.load(state_file)   # shape: (num_trajectories, traj_length, state_dim)
    actions = torch.load(action_file)   # shape: (num_trajectories, traj_length, action_dim)
    
    # Convert to numpy arrays.
    states_np = states.numpy()[:,:,:27]
    actions_np = actions.numpy()
    
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
        expert_data: dict containing keys 'state', 'action', 'reward', 'done', 'next_state'
        device: torch device
        """
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
    model_name = 'Ant'
    env = gym.make(model_name+"-v4", render_mode='human')
    env.reset_model()
    env.reset()

    # print the observation space and action space
    print("observation space:", env.observation_space)
    print("observation space shape:", env.observation_space.shape)
    print("action space:", env.action_space)
    print("action space shape:", env.action_space.shape)

    actions = torch.load(model_name+'_actions.pt').numpy()[1]

    # for i in range(1000):
    done = False
    i = 0
    total_reward = 0
    while not done:
        action = actions[i]
        obs, reward, done, _, _=env.step(action)
        env.render()
        i += 1
        total_reward += reward
        print("Step:", i, "Reward:", total_reward, "Done:", done)
    env.close()