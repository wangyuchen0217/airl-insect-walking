import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.normalized_env import CoppeliaSimEnv


ENV_ID = "Medauroidea_60000_offset"
ALGO = "airl_logit"
FILENAME = "20250819-1650" 
STEP_NUM = 1250000 

STATES_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_1_states.csv"
ACTIONS_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_1_actions.csv"

states = pd.read_csv(STATES_PATH, header=None, index_col=None)
states.columns = [
                  'body_x', 'body_y',
                  'body_z', 'body_roll', 'body_pitch', 'body_yaw', 
                  'LF_ThC', 'LF_CTr', 'LF_FTi', 'LM_ThC', 'LM_CTr', 'LM_FTi', 'LH_ThC', 'LH_CTr', 'LH_FTi',
                  'RF_ThC', 'RF_CTr', 'RF_FTi', 'RM_ThC', 'RM_CTr', 'RM_FTi', 'RH_ThC', 'RH_CTr', 'RH_FTi',
                  'force_LF', 'force_LM', 'force_LH', 'force_RF', 'force_RM', 'force_RH',
                  'foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH'
                  ]
actions = pd.read_csv(ACTIONS_PATH, header=None, index_col=None)
actions.columns = ['LF_ThC', 'LF_CTr', 'LF_FTi', 'LM_ThC', 'LM_CTr', 'LM_FTi', 'LH_ThC', 'LH_CTr', 'LH_FTi',
                                     'RF_ThC', 'RF_CTr', 'RF_FTi', 'RM_ThC', 'RM_CTr', 'RM_FTi', 'RH_ThC', 'RH_CTr', 'RH_FTi']

# denormalize
# env = CoppeliaSimEnv(port=23000, OnTimeStep=True)
# actions = env.denormalize_action(actions)
# convert to degrees
# actions = np.rad2deg(actions)

start = 200
end = 300

# Plot the ThC joint angles [:,0:6] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
ThC = states[['LF_ThC', 'LM_ThC', 'LH_ThC', 'RF_ThC', 'RM_ThC', 'RH_ThC']].values
labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
for i in range(6):
    axs[i].plot(ThC[start:end, i], label=labels[i], color='black')
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('ThC Joint (deg)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.show()

# Plot the CTr joint angles [:,0:6] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
CTr = states[['LF_CTr', 'LM_CTr', 'LH_CTr', 'RF_CTr', 'RM_CTr', 'RH_CTr']].values
labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
for i in range(6):
    axs[i].plot(CTr[start:end, i], label=labels[i], color='black')
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('CTr Joint (deg)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.show()

# Plot the FTi joint angles [:,0:6] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
FTi = states[['LF_FTi', 'LM_FTi', 'LH_FTi', 'RF_FTi', 'RM_FTi', 'RH_FTi']].values
labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
for i in range(6):
    axs[i].plot(FTi[start:end, i], label=labels[i], color='black')
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('FTi Joint (deg)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.show()

# Plot the foot trajectory [:,0:6] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
foot_traj = states[['foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH']].values
labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
for i in range(6):
    axs[i].plot(foot_traj[start:end, i], label=labels[i], color='black')
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('Foot Trajectory (m)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.show()