import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
from envs import *
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import envs
import gymnasium as gym
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
    env_id = 'StickInsect-v0'
    env = gym.make(env_id, render_mode="human")
    env.reset()

    total_reward = 0.0
    for i, angles_deg in enumerate(joint_movement):
        action = np.deg2rad(angles_deg.astype(np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f" Episode ended at step {i}")
            break

    print("Total reward:", total_reward)
    env.close()
    
ANIMAL = "Carausius"
DATA_FILE_1 = "Animal12_110415_00_22.csv"
DATA_FILE_2 = "Animal12_110415_00_23.csv"
DATA_FILE_3 = "Animal12_110415_00_32.csv"

print(ANIMAL, ":", DATA_FILE_1)
joint_movement_1 = joint_prepration(ANIMAL, DATA_FILE_1) 
expert_simulation(joint_movement_1)
print(ANIMAL, ":", DATA_FILE_2)
joint_movement_2 = joint_prepration(ANIMAL, DATA_FILE_2)
expert_simulation(joint_movement_2)
print(ANIMAL, ":", DATA_FILE_3)
joint_movement_3 = joint_prepration(ANIMAL, DATA_FILE_3)
expert_simulation(joint_movement_3)