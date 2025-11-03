import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from common.normalized_env_66k import CoppeliaSimEnv
# from common.normalized_env_66k_legloss import CoppeliaSimEnv
from common.normalized_env_66k_RM_error import CoppeliaSimEnv
import os

# ======== Parameters (modify these as needed) ========= #
ENV_ID = "Medauroidea_66k_aug3c_uneven_legloss"
ALGO = "airl_logit_vx"
FILENAME = "20251022-1712" 
STEP_NUM = "950000"  
EPISODE = 1 # 
start = 0 # 
end = 310 #
fig_length = 10
fig_width = 2.5

STATES_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_states.csv"
ACTIONS_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_actions.csv"
VEL_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_velocities.csv"
FOOT_TRAJ_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_foot_trajs.csv"
'''-----------------------------------need to adjust for the leg loss-----------------------------------'''
RH_PATH = f"logs/{ENV_ID}/{ALGO}/{FILENAME}/eval/step{STEP_NUM}/episode_{EPISODE}_RH_joints.csv"

os.makedirs(f"evaluation/{ENV_ID}/{ALGO}/{FILENAME}/step{STEP_NUM}/episode_{EPISODE}/", exist_ok=True)
SAVE_PATH = f"evaluation/{ENV_ID}/{ALGO}/{FILENAME}/step{STEP_NUM}/episode_{EPISODE}"

# ======== Read Data ======== #
states = pd.read_csv(STATES_PATH, header=None, index_col=None)
actions = pd.read_csv(ACTIONS_PATH, header=None, index_col=None)
'''-----------------------------------need to adjust for the leg loss-----------------------------------'''
actions.columns = [
                                    'LF_ThC', 'LF_CTr', 'LF_FTi', 
                                    'LM_ThC', 'LM_CTr', 'LM_FTi', 
                                    'LH_ThC', 'LH_CTr', 'LH_FTi',
                                    'RF_ThC', 'RF_CTr', 'RF_FTi', 
                                    # 'RM_ThC', 'RM_CTr', 'RM_FTi', 
                                    'RH_ThC', 'RH_CTr', 'RH_FTi'
                                    ]
velocities = pd.read_csv(VEL_PATH, header=None, index_col=None)
foot_trajs = pd.read_csv(FOOT_TRAJ_PATH, header=None, index_col=None)

'''-----------------------------------need to adjust for the leg loss-----------------------------------'''
rh = pd.read_csv(RH_PATH, header=None, index_col=None)
rh.columns = ['RH_ThC', 'RH_CTr', 'RH_FTi']

# ======== Denormalize Data ======== #
env = CoppeliaSimEnv(simulation = False)
states = pd.DataFrame(env.denormalize_observation(states))
actions = env.denormalize_action(actions)

'''-----------------------------------need to adjust for the leg loss-----------------------------------'''
states.columns = [
                #   'body_x', 'body_y',
                  'body_z', 
                  'body_roll', 'body_pitch', 'body_yaw', 
                  'LF_ThC', 'LF_CTr', 'LF_FTi', 
                  'LM_ThC', 'LM_CTr', 'LM_FTi', 
                  'LH_ThC', 'LH_CTr', 'LH_FTi',
                  'RF_ThC', 'RF_CTr', 'RF_FTi', 
                  'RM_ThC', 'RM_CTr', 'RM_FTi', 
                #   'RH_ThC', 'RH_CTr', 'RH_FTi',
                  'force_LF', 'force_LM', 'force_LH', 'force_RF', 'force_RM', 'force_RH',
                #   'foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH',
                #   'contact_FL', 'contact_ML', 'contact_HL', 'contact_FR', 'contact_MR', 'contact_HR'
                  ]

'''-----------------------------------need to adjust for the leg loss-----------------------------------'''
states['RH_ThC'] = rh['RH_ThC']
states['RH_CTr'] = rh['RH_CTr']
states['RH_FTi'] = rh['RH_FTi']

# ======== Plotting Functions ======== #
def plot_6_legs(policy, expert, variable_name, start, end, title):
    fig, axs = plt.subplots(6, 1, figsize=(fig_length, fig_width))
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
    plt.savefig(os.path.join(SAVE_PATH, f"{title}.png"))


def plot_1_joint(policy, joint_label, start, end, title):
    plt.figure(figsize=(fig_length, fig_width))
    leg_labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
    for leg in leg_labels:
        label = f"{leg}_{joint_label}"
        plt.plot(policy[label][start:end], label=leg)
    plt.title(title, fontsize=18)
    plt.xlabel('Time (frames)', fontsize=16)
    plt.ylabel('Joint Angle (rad)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{title}.png"))


def plot_pose(policy, pose_name, start, end, title):
    plt.figure(figsize=(fig_length, fig_width))
    for pose in pose_name:
        plt.plot(policy[pose][start:end], label=f"Policy {pose}")
    plt.title(title, fontsize=18)
    plt.xlabel('Time (frames)', fontsize=16)
    plt.ylabel('Angle (rad)', fontsize=16)
    plt.xlim(start, end)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{title}.png"))


def plot_gait(policy, foot_trajs, start, end, title):
    force_cols = ['force_LF', 'force_LM', 'force_LH', 'force_RF', 'force_RM', 'force_RH']
    foot_cols = ['foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH']
    leg_labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    def get_contact_byforce(data):
        """Compute binary contact states (1=stance, 0=swing)."""
        contact = np.zeros((len(data), len(force_cols)))
        for i, col in enumerate(force_cols):
            contact[:, i] = (data[col].abs() > 0.27).astype(int)
        return contact[start:end]

    def get_contact_byfoot(data):
        """Compute binary contact states (1=stance, 0=swing)."""
        contact = np.zeros((len(data), len(foot_cols)))
        for i, col in enumerate(foot_cols):
            contact[:, i] = (data[i].abs() < 0.02).astype(int)
        return contact[start:end]
    
    contact_policy = get_contact_byforce(policy)
    contact_foot_trajs = get_contact_byfoot(foot_trajs)

    plt.figure(figsize=(fig_length, fig_width))
    plt.imshow(contact_policy.T, aspect="auto", cmap="Greys", interpolation="nearest")
    plt.title("Policy Gait", fontsize=18)
    plt.xlabel("Time (frames)", fontsize=16)
    plt.ylabel("Legs", fontsize=16)
    plt.yticks(range(len(leg_labels)), leg_labels, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{title} Force.png"))

    plt.figure(figsize=(fig_length, fig_width))
    plt.imshow(contact_foot_trajs.T, aspect="auto", cmap="Greys", interpolation="nearest")
    plt.title("Policy Gait", fontsize=18)
    plt.xlabel("Time (frames)", fontsize=16)
    plt.ylabel("Legs", fontsize=16)
    plt.yticks(range(len(leg_labels)), leg_labels, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{title} Foot Trajs.png"))


def plot_vel(policy, start, end, title):
    plt.figure(figsize=(fig_length, fig_width))
    plt.plot(policy[start:end])
    plt.title(title, fontsize=18)
    plt.xlabel('Time (frames)', fontsize=16)
    plt.ylabel('Vel (cm/s)', fontsize=16)
    plt.xlim(start, end)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{title}.png"))


def plot_foot_trajs(policy, start, end, title):
    '''-----------------------------------need to adjust for the leg loss-----------------------------------'''
    foot_trajs.iloc[:, 4] = 0
    fig, axs = plt.subplots(6, 1, figsize=(fig_length, fig_width*3))
    foot_labels = ['foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH']
    for i, foot in enumerate(foot_labels):
        axs[i].plot(policy[i][start:end])
        axs[i].set_ylabel(foot[-2:], fontsize=16)
        axs[i].tick_params(axis='both', which='major', labelsize=14)
        axs[i].grid()
    axs[0].set_title('Foot Trajectory (m)', fontsize=18)
    axs[-1].set_xlabel('Time (frames)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{title}.png"))

# ======== Generate Plots ======== #

# plot_6_legs(states, expert_states, ['foot_traj_LF', 'foot_traj_LM', 'foot_traj_LH', 'foot_traj_RF', 'foot_traj_RM', 'foot_traj_RH'], start, end, 'State: Foot Trajectory')

plot_1_joint(states, 'ThC', start, end, 'Joints ThC')
plot_1_joint(states, 'CTr', start, end, 'Joints CTr')
plot_1_joint(states, 'FTi', start, end, 'Joints FTi')

plot_pose(states, ['body_roll', 'body_pitch', 'body_yaw'], start, end, 'Body Pose')
plot_gait(states, foot_trajs, start, end, title="Gait Pattern")

plot_vel(velocities, start, end, title="Velocity")
plot_foot_trajs(foot_trajs, start, end, title="Foot Trajectories")
