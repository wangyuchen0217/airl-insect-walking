import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.normalized_env import CoppeliaSimEnv
import os

# ======== Parameters (modify these as needed) ========= #
ENV_ID = "Medauroidea_60000_offset"
ALGO = "airl_logit"
FILENAME = "20250819-1650" 
STEP_NUM = 1250000 
EPISODE = 1
start = 200
end = 300

STATES_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_states.csv"
ACTIONS_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_actions.csv"

os.makedirs(f"evaluation/{ENV_ID}/{ALGO}/{FILENAME}/step{STEP_NUM}/episode_{EPISODE}/", exist_ok=True)
SAVE_PATH = f"evaluation/{ENV_ID}/{ALGO}/{FILENAME}/step{STEP_NUM}/episode_{EPISODE}"

EXPERT_STATES_PATH = "evaluation/expert_states_normalized.csv"
EXPERT_ACTION_PATH = "evaluation/expert_actions_normalized.csv"

# ======== Read Data ======== #
states = pd.read_csv(STATES_PATH, header=None, index_col=None)
actions = pd.read_csv(ACTIONS_PATH, header=None, index_col=None)
actions.columns = ['LF_ThC', 'LF_CTr', 'LF_FTi', 'LM_ThC', 'LM_CTr', 'LM_FTi', 'LH_ThC', 'LH_CTr', 'LH_FTi',
                                     'RF_ThC', 'RF_CTr', 'RF_FTi', 'RM_ThC', 'RM_CTr', 'RM_FTi', 'RH_ThC', 'RH_CTr', 'RH_FTi']

expert_states = pd.read_csv(EXPERT_STATES_PATH, header=None, index_col=None)
expert_actions = pd.read_csv(EXPERT_ACTION_PATH, header=None, index_col=None)
expert_actions.columns = actions.columns

# ======== Denormalize Data ======== #
env = CoppeliaSimEnv(simulation = False)
states = pd.DataFrame(env.denormalize_observation(states))
actions = env.denormalize_action(actions)
expert_states = pd.DataFrame(env.denormalize_observation(expert_states))
expert_actions = env.denormalize_action(expert_actions)

states.columns = [
                  'body_x', 'body_y',
                  'body_z', 'body_roll', 'body_pitch', 'body_yaw', 
                  'LF_ThC', 'LF_CTr', 'LF_FTi', 'LM_ThC', 'LM_CTr', 'LM_FTi', 'LH_ThC', 'LH_CTr', 'LH_FTi',
                  'RF_ThC', 'RF_CTr', 'RF_FTi', 'RM_ThC', 'RM_CTr', 'RM_FTi', 'RH_ThC', 'RH_CTr', 'RH_FTi',
                  'force_LF', 'force_LM', 'force_LH', 'force_RF', 'force_RM', 'force_RH',
                  'foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH'
                  ]
expert_states.columns = states.columns

# ======== Plotting Functions ======== #
def plot_6_legs(policy, expert, variable_name, start, end, title):
    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    policy_data = policy[variable_name].values
    expert_data = expert[variable_name].values
    ylabel = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
    for i in range(6):
        axs[i].plot(policy_data[start:end, i], label="policy")
        axs[i].plot(expert_data[start:end, i], label="expert")
        axs[i].set_ylabel(ylabel[i], fontsize=16)
        axs[i].tick_params(axis='both', which='major', labelsize=14)
        axs[i].grid()
    axs[0].set_title(title, fontsize=18)
    axs[-1].set_xlabel('Time (frames)', fontsize=16)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, ncol=1, loc='outside center right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(SAVE_PATH, f"{title }.png"))

def plot_1_joint(policy, expert, joint_label, start, end, title):
    fig, axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    leg_labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    for i, leg in enumerate(leg_labels):
        label = f"{leg}_{joint_label}"
        axs[0].plot(policy[label][start:end], label=leg)
        axs[1].plot(expert[label][start:end], label=leg)
        axs[0].set_ylabel('Policy', fontsize=16)
        axs[1].set_ylabel('Expert', fontsize=16)
    axs[0].set_title(title, fontsize=18)
    axs[1].set_xlabel('Time (frames)', fontsize=16)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, ncol=1, loc='outside center right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(SAVE_PATH, f"{title }.png"))

def plot_pose(policy, expert, pose_name, start, end, title):
    fig, axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    for i, pose in enumerate(pose_name):
        axs[0].plot(policy[pose][start:end], label=pose)
        axs[1].plot(expert[pose][start:end], label=pose)
        axs[0].set_ylabel('Policy', fontsize=16)
        axs[1].set_ylabel('Expert', fontsize=16)
    axs[0].set_title(title, fontsize=18)
    axs[1].set_xlabel('Time (frames)', fontsize=16)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, ncol=1, loc='outside center right')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(os.path.join(SAVE_PATH, f"{title }.png"))

# ======== Generate Plots ======== #
plot_6_legs(actions, expert_actions, ['LF_ThC', 'LM_ThC', 'LH_ThC', 'RF_ThC', 'RM_ThC', 'RH_ThC'], start, end, 'Action: ThC Joint')
plot_6_legs(states, expert_states, ['LF_ThC', 'LM_ThC', 'LH_ThC', 'RF_ThC', 'RM_ThC', 'RH_ThC'], start, end, 'State: ThC Joint')

plot_6_legs(actions, expert_actions, ['LF_CTr', 'LM_CTr', 'LH_CTr', 'RF_CTr', 'RM_CTr', 'RH_CTr'], start, end, 'Action: CTr Joint')
plot_6_legs(states, expert_states, ['LF_CTr', 'LM_CTr', 'LH_CTr', 'RF_CTr', 'RM_CTr', 'RH_CTr'], start, end, 'State: CTr Joint')

plot_6_legs(actions, expert_actions, ['LF_FTi', 'LM_FTi', 'LH_FTi', 'RF_FTi', 'RM_FTi', 'RH_FTi'], start, end, 'Action: FTi Joint')
plot_6_legs(states, expert_states, ['LF_FTi', 'LM_FTi', 'LH_FTi', 'RF_FTi', 'RM_FTi', 'RH_FTi'], start, end, 'State: FTi Joint')

plot_6_legs(states, expert_states, ['foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH'], start, end, 'State: Foot Trajectory')

plot_1_joint(states, expert_states, 'ThC', start, end, 'Joints ThC')
plot_1_joint(states, expert_states, 'CTr', start, end, 'Joints CTr')
plot_1_joint(states, expert_states, 'FTi', start, end, 'Joints FTi')

plot_pose(states, expert_states, ['body_roll', 'body_pitch', 'body_yaw'], start, end, 'State: Body Pose')