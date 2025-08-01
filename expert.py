import torch
import numpy as np
import pandas as pd

def load_expert_data(expert_file, save_npz=False, npz_filename="expert_data.npz"):

    # load the expert data (CoppeliaSim)
    data = pd.read_csv(expert_file, header=[0])
    states_np = data[['body_z', 'body_roll', 'body_pitch', 'body_yaw', 
                  'motor_pos_FL_TC', 'motor_pos_FL_CF', 'motor_pos_FL_FT', 
                  'motor_pos_ML_TC', 'motor_pos_ML_CF', 'motor_pos_ML_FT',
                  'motor_pos_HL_TC', 'motor_pos_HL_CF', 'motor_pos_HL_FT',
                  'motor_pos_FR_TC', 'motor_pos_FR_CF', 'motor_pos_FR_FT',
                  'motor_pos_MR_TC', 'motor_pos_MR_CF', 'motor_pos_MR_FT',
                  'motor_pos_HR_TC', 'motor_pos_HR_CF', 'motor_pos_HR_FT',
                  'force_FL', 'force_ML', 'force_HL', 'force_FR', 'force_MR', 'force_HR',
                  'FL_foot_traj_z', 'ML_foot_traj_z', 'HL_foot_traj_z',
                  'FR_foot_traj_z', 'MR_foot_traj_z', 'HR_foot_traj_z']].values
    actions_np = data[['motor_cmd_FL_TC', 'motor_cmd_FL_CF', 'motor_cmd_FL_FT',
                   'motor_cmd_ML_TC', 'motor_cmd_ML_CF', 'motor_cmd_ML_FT',
                  'motor_cmd_HL_TC', 'motor_cmd_HL_CF', 'motor_cmd_HL_FT',
                  'motor_cmd_FR_TC', 'motor_cmd_FR_CF', 'motor_cmd_FR_FT',
                  'motor_cmd_MR_TC', 'motor_cmd_MR_CF', 'motor_cmd_MR_FT',
                  'motor_cmd_HR_TC', 'motor_cmd_HR_CF', 'motor_cmd_HR_FT']].values
    # print(f"States shape: {states.shape}, Actions shape: {actions.shape}")

    states = []
    actions = []
    next_states = []
    for i in range(len(states_np) - 1):
        states.append(states_np[i])
        actions.append(actions_np[i])
        next_states.append(states_np[i + 1])

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    rewards = np.zeros(len(states), dtype=np.float32)  # Assuming zero rewards for expert data (not affect AIRL training)
    dones = np.zeros(len(states), dtype=np.float32)  # Assuming no terminal states for expert data

    expert_data = {
        'state': states,
        'action': actions,
        'reward': rewards,
        'done': dones,
        'next_state': next_states
    }
    # print(f"Expert data states: {expert_data['state'].shape}, actions: {expert_data['action'].shape}")
    # print(expert_data)

    if save_npz:
        np.savez(npz_filename, **expert_data)

    return expert_data


def load_expert_cutlegs_data(expert_file, save_npz=False, npz_filename="expert_data.npz"):

    # load the expert data (CoppeliaSim)
    data = pd.read_csv(expert_file, header=[0])
    states_np = data[['body_z', 'body_roll', 'body_pitch', 'body_yaw', 
                  'motor_pos_ML_TC', 'motor_pos_ML_CF', 'motor_pos_ML_FT',
                  'motor_pos_HL_TC', 'motor_pos_HL_CF', 'motor_pos_HL_FT',
                  'motor_pos_MR_TC', 'motor_pos_MR_CF', 'motor_pos_MR_FT',
                  'motor_pos_HR_TC', 'motor_pos_HR_CF', 'motor_pos_HR_FT',
                  'force_ML', 'force_HL', 'force_MR', 'force_HR',
                  'ML_foot_traj_z', 'HL_foot_traj_z',
                  'MR_foot_traj_z', 'HR_foot_traj_z']].values
    actions_np = data[['motor_cmd_ML_TC', 'motor_cmd_ML_CF', 'motor_cmd_ML_FT',
                  'motor_cmd_HL_TC', 'motor_cmd_HL_CF', 'motor_cmd_HL_FT',
                  'motor_cmd_MR_TC', 'motor_cmd_MR_CF', 'motor_cmd_MR_FT',
                  'motor_cmd_HR_TC', 'motor_cmd_HR_CF', 'motor_cmd_HR_FT']].values
    # print(f"States shape: {states.shape}, Actions shape: {actions.shape}")

    states = []
    actions = []
    next_states = []
    for i in range(len(states_np) - 1):
        states.append(states_np[i])
        actions.append(actions_np[i])
        next_states.append(states_np[i + 1])

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    rewards = np.zeros(len(states), dtype=np.float32)  # Assuming zero rewards for expert data (not affect AIRL training)
    dones = np.zeros(len(states), dtype=np.float32)  # Assuming no terminal states for expert data

    expert_data = {
        'state': states,
        'action': actions,
        'reward': rewards,
        'done': dones,
        'next_state': next_states
    }
    # print(f"Expert data states: {expert_data['state'].shape}, actions: {expert_data['action'].shape}")
    # print(expert_data)

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

    EXPERT_FILE = "expert.csv"  # Path to the expert data CSV file
    expert_data = load_expert_cutlegs_data(EXPERT_FILE, save_npz=False, npz_filename="expert_data.npz")

    action_max = np.max(expert_data['action'], axis=0)
    action_min = np.min(expert_data['action'], axis=0)
    print(f"Action max: {action_max}, Action min: {action_min}")

    state_max = np.max(expert_data['state'], axis=0)
    state_min = np.min(expert_data['state'], axis=0)
    print(f"State max: {state_max}, State min: {state_min}")
