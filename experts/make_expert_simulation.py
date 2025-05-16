import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
from envs import *
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import torch


# smooth the data
def Kalman1D(observations,damping=1):
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.03
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
            )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def data_smooth(data):
    for i in range(data.shape[1]):
        smoothed_data = Kalman1D(data[:,i], damping=1).reshape(-1,1)
        data[:,i] = smoothed_data[:,0]
    return data

def joint_prepration(ANIMAL, DATA_FILE):
    # Load the joint angle data
    joint_path = os.path.join("experts/stickinsects", ANIMAL, DATA_FILE)
    joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
    joint_movement = data_smooth(joint_movement) # smooth the data

    # FTi joint angle minus 90 degree
    joint_movement[:,-6:] = joint_movement[:,-6:] - 90
    # remove the sup data
    joint_movement = joint_movement[:,6:]
    print("joint_movement:", joint_movement.shape)

    return joint_movement

def expert_simulation(joint_movement):
    #  Set up simulation without rendering
    model_name = 'StickInsect-v5'
    model_path = 'envs/assets/' + model_name + '.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    obs_state = []
    leg_geoms = ['LF_tibia_geom', 'LM_tibia_geom', 'LH_tibia_geom', 'RF_tibia_geom', 'RM_tibia_geom', 'RH_tibia_geom']
    leg_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in leg_geoms]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    contact_matrix = np.zeros((joint_movement.shape[0], len(leg_geoms)), dtype=int)
    force_data = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # set a camera <camera name="top" mode="fixed" pos="5 0 20" xyaxes="1 0 0 0 1 0"/>
        viewer.cam.lookat[0] = 5  # x-coordinate of the point to look at
        viewer.cam.lookat[1] = 0  # y-coordinate
        viewer.cam.lookat[2] = 0  # z-coordinate
        viewer.cam.distance = 20  # Camera distance from the lookat point
        viewer.cam.azimuth = 90  # Camera azimuth angle in degrees
        viewer.cam.elevation = -90  # Camera elevation angle in degrees

        for j in range(joint_movement.shape[0]):  # Run the simulation for the length of the joint movement data
            if not viewer.is_running():  # Check if the viewer has been closed manually
                break
            # implement the joint angle data
            joint_angle = np.deg2rad(joint_movement[j])
            data.ctrl = joint_angle
            mujoco.mj_step(model, data)
            viewer.sync()
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            # Manage timing to maintain a steady frame rate
            time.sleep(model.opt.timestep)

            state = np.hstack((data.qpos.copy()[2:], # [2:] remove the root position
                                                data.qvel.copy()))
            # record the state of each step
            obs_state.append(state)
            
            # record contact data
            for i in range(data.ncon):
                    contact = data.contact[i]
                    geom1 = contact.geom1
                    geom2 = contact.geom2
                    # Check if the contact involves a leg geom and the floor
                    for leg_index, leg_id in enumerate(leg_ids):
                        if (geom1 == leg_id and geom2 == floor_id) or (geom1 == floor_id and geom2 == leg_id):
                            contact_matrix[j, leg_index] = 1  # Mark contact

            # record the force sensor data
            sensor_data = data.sensordata[6:].copy()
            force_data.append(sensor_data)

    # record observation state and action
    obs_states = np.array(obs_state) # [len, 47] with torso without root position
    print("states:", obs_states.shape)
    actions = np.array((np.deg2rad(joint_movement))-obs_states[:, 5:23]) 
    print("actions:", actions.shape)  # [len, 18]
    contact_matrix = np.array(contact_matrix) # [len, 6]
    print("contact_matrix:", contact_matrix.shape)
    force_data = np.array(force_data) # [len, 18]
    print("force_data:", force_data.shape) # [len, 18]
    return obs_states, actions, contact_matrix, force_data

def plot_contact_gait(contact_matrix):
    plt.figure(figsize=(7, 6))
    for leg in range(contact_matrix.shape[1]):
        plt.fill_between(range(contact_matrix.shape[0]), 
                        leg * 1.5, leg * 1.5 + 1, 
                        where=contact_matrix[:, leg] == 1, 
                        color='black', step='mid')
    plt.yticks([leg * 1.5 + 0.5 for leg in range(6)], ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']) 
    plt.gca().invert_yaxis()
    plt.xlabel('Time Step')
    plt.title('Gait Phase Plot kp300kv200')
    plt.show()

def plot_expert_demo(obs_states, actions):
    idx_j = 0 # 0--17 joint angles
    idx_v= 18 # 18--35 joint velocities
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(obs_states[:, idx_j+7], label="states_qpos", color="blue")
    axs[0].set_title("joint position states")
    axs[1].plot(obs_states[:, idx_v+13], label="states_qvel", color="blue")
    axs[1].set_title("joint velocity states")
    axs[2].plot(actions[:, idx_j], label="actions", color="red")
    axs[2].set_title("actions")
    plt.show()

ANIMAL = "Carausius"
DATA_FILE_1 = "Animal12_110415_00_22.csv"
DATA_FILE_2 = "Animal12_110415_00_23.csv"
DATA_FILE_3 = "Animal12_110415_00_32.csv"

# print(ANIMAL, ":", DATA_FILE_1)
# joint_movement_1 = joint_prepration(ANIMAL, DATA_FILE_1) 
# obs_states_1, actions_1, contact_matrix_1, force_1 = expert_simulation(joint_movement_1)
print(ANIMAL, ":", DATA_FILE_2)
joint_movement_2 = joint_prepration(ANIMAL, DATA_FILE_2)
obs_states_2, actions_2, contact_matrix_2, force_2 = expert_simulation(joint_movement_2)
print(ANIMAL, ":", DATA_FILE_3)
joint_movement_3 = joint_prepration(ANIMAL, DATA_FILE_3)
obs_states_3, actions_3, contact_matrix_3, force_3 = expert_simulation(joint_movement_3)

# # save force data as csv
# np.savetxt("experts/StickInsect_force_1.csv", force_1, delimiter=",")
# np.savetxt("experts/StickInsect_force_2.csv", force_2, delimiter=",")
# np.savetxt("experts/StickInsect_force_3.csv", force_3, delimiter=",")

expert_states = np.concatenate((obs_states_2, obs_states_3), axis=0)
expert_actions = np.concatenate((actions_2, actions_3), axis=0)
print("---")
print("expert states:", expert_states.shape)
print("expert actions:", expert_actions.shape)

# save as csv
# np.savetxt("experts/StickInsect_states_v2.csv", expert_states, delimiter=",")
# np.savetxt("experts/StickInsect_actions_v2.csv", expert_actions, delimiter=",")

# save numpy data as pt file
expert_states = torch.tensor(expert_states, dtype=torch.float32)
expert_actions = torch.tensor(expert_actions, dtype=torch.float32)
torch.save(expert_states, "experts/StickInsect_states_v3.pt")
torch.save(expert_actions, "experts/StickInsect_actions_v3.pt")

