'''
This script provides the module for loading expert demonstration data from the original CoppeliaSim CSV file format.
It includes functions to load the expert data, add contact columns based on force thresholds, and create a symmetric action bounds for left and right legs.
'''


import torch
import numpy as np
import pandas as pd

def load_expert_data(expert_file, save_npz=False, npz_filename="expert_data.csv"):

    # load the expert data (CoppeliaSim)
    data = pd.read_csv(expert_file, header=[0])
    states_np = data[[
                                    # 'body_x', 'body_y',
                                    # 'body_z', 
                                    'body_roll', 'body_pitch', 'body_yaw', 
                                    'motor_pos_FL_TC', 'motor_pos_FL_CF', 'motor_pos_FL_FT', 
                                    'motor_pos_ML_TC', 'motor_pos_ML_CF', 'motor_pos_ML_FT',
                                    'motor_pos_HL_TC', 'motor_pos_HL_CF', 'motor_pos_HL_FT',
                                    'motor_pos_FR_TC', 'motor_pos_FR_CF', 'motor_pos_FR_FT',
                                    'motor_pos_MR_TC', 'motor_pos_MR_CF', 'motor_pos_MR_FT',
                                    'motor_pos_HR_TC', 'motor_pos_HR_CF', 'motor_pos_HR_FT',
                                    # 'qvel_body_x', 'qvel_body_y', 'qvel_body_z', 'qvel_body_roll', 'qvel_body_pitch', 'qvel_body_yaw',
                                    # 'qvel_FL_TC', 'qvel_FL_CF', 'qvel_FL_FT',
                                    # 'qvel_ML_TC', 'qvel_ML_CF', 'qvel_ML_FT',
                                    # 'qvel_HL_TC', 'qvel_HL_CF', 'qvel_HL_FT',
                                    # 'qvel_FR_TC', 'qvel_FR_CF', 'qvel_FR_FT',
                                    # 'qvel_MR_TC', 'qvel_MR_CF', 'qvel_MR_FT',
                                    # 'qvel_HR_TC', 'qvel_HR_CF', 'qvel_HR_FT',
                                    # 'force_FL', 'force_ML', 'force_HL', 'force_FR', 'force_MR', 'force_HR',
                                    # 'FL_foot_traj_z', 'ML_foot_traj_z', 'HL_foot_traj_z', 'FR_foot_traj_z', 'MR_foot_traj_z', 'HR_foot_traj_z',
                                    'contact_FL', 'contact_ML', 'contact_HL', 'contact_FR', 'contact_MR', 'contact_HR'
                                ]].values
    actions_np = data[[
                                    'motor_cmd_FL_TC', 'motor_cmd_FL_CF', 'motor_cmd_FL_FT',
                                    'motor_cmd_ML_TC', 'motor_cmd_ML_CF', 'motor_cmd_ML_FT',
                                    'motor_cmd_HL_TC', 'motor_cmd_HL_CF', 'motor_cmd_HL_FT',
                                    'motor_cmd_FR_TC', 'motor_cmd_FR_CF', 'motor_cmd_FR_FT',
                                    'motor_cmd_MR_TC', 'motor_cmd_MR_CF', 'motor_cmd_MR_FT',
                                    'motor_cmd_HR_TC', 'motor_cmd_HR_CF', 'motor_cmd_HR_FT'
                                ]].values

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

    # Remove the inital position offset
    init_pos_deg = np.array([[30, 9.5, -60], 
                                                        [ 0 ,  -2.5, -60],
                                                        [-40, 9.5,-60],
                                                        [30, 9.5, -60], 
                                                        [0, -2.5, -60],
                                                        [-40, 9.5, -60]], dtype=float).astype(float)  # initial joint position in degrees
    init_pos_dirction = np.array([[-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [1, -1, -1],
                                                            [1, -1, -1],
                                                            [1, -1, -1]])  
    init_pos_deg = init_pos_deg * init_pos_dirction  # adjust the initial position direction
    init_pos_rad = np.deg2rad(init_pos_deg).flatten()  # convert to radians
    actions = actions - init_pos_rad  # remove the initial position offset

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


def add_contact_columns(expert_file, save = False, save_file="expert_60000_with_contact.csv"):
    # load the expert data (CoppeliaSim)
    data = pd.read_csv(expert_file, header=[0])

    legs = ['FL', 'ML', 'HL', 'FR', 'MR', 'HR']
    for leg in legs:
        force_col = f'force_{leg}'
        # foot_col = f'{leg}_foot_traj_z'
        contact_col = f'contact_{leg}'
        # if force is not zero, contact is 1, else 0
        data[contact_col] = (data[force_col].abs() > 0.27).astype(int)
        # # if foot height is less than 0.02, contact is 1, else 0
        # data[contact_col] = (data[foot_col].abs() < 0.02).astype(int)
    
    if save:
        data.to_csv(save_file, index=False)

def symmetric_lr_bounds(low_1d: np.ndarray,
                        high_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Make left-right symmetric bounds for 6 legs × 3 joints (TC, CF, FT) in the order:
      [FL_TC, FL_CF, FL_FT,
       ML_TC, ML_CF, ML_FT,
       HL_TC, HL_CF, HL_FT,
       FR_TC, FR_CF, FR_FT,
       MR_TC, MR_CF, MR_FT,
       HR_TC, HR_CF, HR_FT]

    Returns:
        new_low_1d, new_high_1d with the same shape/order as inputs,
        where each L/R pair shares the same outer (min-of-mins / max-of-maxes) bounds
        under the canonical sign defined by init_pos_direction. No ±M symmetry is enforced.
    """
    init_pos_direction = np.array([[-1, 1, 1],   # FL
                                                                [-1, 1, 1],   # ML
                                                                [-1, 1, 1],   # HL
                                                                [ 1,-1,-1],   # FR
                                                                [ 1,-1,-1],   # MR
                                                                [ 1,-1,-1]], dtype=float)   # HR
                                        
    low = low_1d.reshape(6, 3).astype(float).copy()
    high = high_1d.reshape(6, 3).astype(float).copy()
    S = init_pos_direction.astype(float)
    assert low.shape == (6,3) and high.shape == (6,3) and S.shape == (6,3)

    # 1) Map to canonical sign space
    low_c = low * S
    high_c = high * S
    # Ensure proper ordering after possible sign flip
    low_c, high_c = np.minimum(low_c, high_c), np.maximum(low_c, high_c)

    # 2) For each L/R pair, take outer bounds in canonical space
    pairs = [(0,3),  # FL ↔ FR
             (1,4),  # ML ↔ MR
             (2,5)]  # HL ↔ HR

    new_low_c  = low_c.copy()
    new_high_c = high_c.copy()

    for iL, iR in pairs:
        # per joint (TC, CF, FT)
        pair_low  = np.minimum(low_c[iL],  low_c[iR])
        pair_high = np.maximum(high_c[iL], high_c[iR])
        new_low_c[iL]  = pair_low
        new_low_c[iR]  = pair_low
        new_high_c[iL] = pair_high
        new_high_c[iR] = pair_high

    # 3) Map back to original sign space and ensure ordering
    new_low  = new_low_c  * S
    new_high = new_high_c * S

    # After sign re-application, enforce low <= high
    final_low  = np.minimum(new_low,  new_high)
    final_high = np.maximum(new_low,  new_high)
    print("Low (Symmetrical) :", final_low.reshape(-1))
    print("High (Symmetrical) :", final_high.reshape(-1))


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

    EXPERT_FILE = "expert/expert_60000.csv"  # Path to the expert data CSV file

    # add contact columns to the expert csv file if needed
    ADD_CONTACT = False
    if ADD_CONTACT:
        add_contact_columns(EXPERT_FILE, save=True, save_file="expert/expert_60000_fcontact.csv")
    
    expert_data = load_expert_data(EXPERT_FILE, save_npz=False, npz_filename="expert_data.npz")

    action_max = np.max(expert_data['action'], axis=0)
    action_min = np.min(expert_data['action'], axis=0)
    print(f"Action max: {action_max}, Action min: {action_min}")

    state_max = np.max(expert_data['state'], axis=0)
    state_min = np.min(expert_data['state'], axis=0)
    print(f"State max: {state_max}, State min: {state_min}")

    # # make the action bounds symmetric for left and right legs
    # symmetric_lr_bounds(action_max, action_min)



    # # plot force and contact
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(expert_data['state'][:100, -7], label='contact_HR')
    # plt.plot(expert_data['state'][:100, -13], label='foot_HR')
    # plt.xlabel('Time step')
    # plt.ylabel('Value')
    # plt.title('Contact and Force over Time')
    # plt.show()

    # # save expert state as csv
    # np.savetxt("expert_states.csv", expert_data['state'], delimiter=',')
    # np.savetxt("expert_actions.csv", expert_data['action'], delimiter=',')
