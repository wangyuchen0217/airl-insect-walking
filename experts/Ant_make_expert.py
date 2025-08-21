import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
from envs import *
import numpy as np
import pandas as pd
import mujoco.viewer
from common.env import make_env
from scipy.signal import savgol_filter

def load_actions(file_path):
    actions = pd.read_csv(file_path, header=[0]).values
    print("Actions loaded:", actions.shape)
    return actions

def make_loopable_cycle(data, window_length=200, polyorder=3, overlap_len=50):
    """
    :param data: (T, D) gait sequence (T time steps, D dimensions)
    :param window_length: smoothing window length (should be odd and > polyorder)
    :param polyorder: smoothing polynomial degree
    :param overlap_len: how many frames to overlap head-tail
    """
    assert len(data.shape) == 2  # (T, D)
    T, D = data.shape

    # Step 1: connect tail and head with some overlap
    head = data[:overlap_len]
    tail = data[-overlap_len:]
    extended = np.concatenate([tail, data, head], axis=0)  # shape: (T + 2 * overlap_len, D)

    # Step 2: smooth the extended trajectory
    smoothed = savgol_filter(extended, window_length=window_length, polyorder=polyorder, axis=0)

    # Step 3: trim back to original length, centered to avoid boundary artifacts
    start = overlap_len
    end = overlap_len + T
    loopable = smoothed[start:end]

    return loopable

def action_repeat(actions, repeat=40):
    repeated_actions = []
    for i in range(repeat):
        if repeated_actions == []:
            repeated_actions = actions
        repeated_actions = np.concatenate((repeated_actions, actions), axis=0)
    repeated_actions = np.array(repeated_actions)
    print("Actions repeated:", repeated_actions.shape)
    return repeated_actions

def expert_simulation(actions):
    #  Set up simulation without rendering
    env_id = 'Ant-v4'
    env = make_env(env_id, test=True)
    env.reset()

    recording_states = []
    recording_actions = []

    # # Set the initial states
    # initial_state = np.array([0.74112136, 0.9937715, -0.10452032, 0.02629053, -0.02832862, 
    #                           0.04364716, -0.02023279, -0.07135721, 0.07398283, 0.04997852,
    #                           -0.06739895, -0.06164495, 0.01545467, -0.12949677, 0.01603563,
    #                           0.09225433, -0.02452037, 0.07183438, -0.12738715, 0.09307833,
    #                           0.02721602, -0.13936335, -0.01876108, 0.01769702, -0.05388882,
    #                           0.01040591, -0.07004377])
    # data = env.unwrapped.data
    # model = env.unwrapped.model
    # nq, nv = model.nq, model.nv
    # print(f"nq: {nq}, nv: {nv}, obs shape: {env.observation_space.shape}")
    # qpos = data.qpos.copy()
    # qvel = data.qvel.copy()
    # # qpos[0], qpos[1] = 0, 0
    # qpos[2:] = initial_state[:(nq - 2)]
    # qvel[:] = initial_state[(nq - 2):]

    # env.unwrapped.set_state(qpos, qvel)

    total_reward = 0.0
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        recording_states.append(obs)
        recording_actions.append(action)
        if terminated or truncated:
            print(f" Episode ended at step {i}")
            break

    print("Total reward:", total_reward)
    env.close()

    return recording_states, recording_actions


#-------------Main Script-------------------#
ACTION_PATH = "experts/ant_make_dataset/actions.csv"

actions = load_actions(ACTION_PATH)
actions = actions[:80]

# collect 800 trials and record the actions and states separately
states_trail = []
actions_trail = []

for i in range(800):
    state, action = expert_simulation(actions)
    states_trail.append(state)
    actions_trail.append(action)

# Reshape the states and actions
states_trail = np.array(states_trail)
actions_trail = np.array(actions_trail)
states_trail = np.concatenate(states_trail, axis=0)
actions_trail = np.concatenate(actions_trail, axis=0)
print("States shape:", states_trail.shape)
print("Actions shape:", actions_trail.shape)

# Save the states and actions to csv files
states_path = "experts/ant_make_dataset/expert_states.csv"
actions_path = "experts/ant_make_dataset/expert_actions.csv"
np.savetxt(states_path, states_trail, delimiter=",")
np.savetxt(actions_path, actions_trail, delimiter=",")


