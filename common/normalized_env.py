'''
This module defines a CoppeliaSim environment class that interfaces with the CoppeliaSim simulator using ZeroMQ.
It provides functionalities for normalizing and denormalizing observations and actions, controlling the robot's joints,
and retrieving simulation data such as joint angles, body position, orientation, forces, foot trajectory, and contact states.
Applicable for: expert/expert.csv (not used for current version)
'''


import zmq
import msgpack
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys

class CoppeliaSimEnv:

    _max_episode_steps = 1000
    _step_count = 0

    __leg_names = ['_FL','_ML','_HL','_FR','_MR','_HR']
    __joint_names = ['/m1', '/m2', '/m3']  # ThC, CTr, FTi
    __foot_names = ['/foot_FL', '/foot_ML', '/foot_HL', 
                    '/foot_FR', '/foot_MR', '/foot_HR']
    __IMU_names = ['/IMU_robot', '/IMU_ref']
    __forcesensor_names = ['/forceSensor_FL', '/forceSensor_ML', '/forceSensor_HL', 
                           '/forceSensor_FR', '/forceSensor_MR', '/forceSensor_HR']

    __joint_handle = np.zeros((6, 3), dtype=int).astype(int)  # joint handle (leg l, joint j)
    __target_positions = np.zeros((6, 3), dtype=float).astype(float)  # joint target position (leg l, joint j)
    __initjoint_position = np.zeros((6, 3), dtype=float).astype(float)  # initial joint position (leg l, joint j)
    __init_pos_deg = np.array([[30, 9.5, -60], 
                                                        [ 0 ,  -2.5, -60],
                                                        [-40, 9.5,-60],
                                                        [30, 9.5, -60], 
                                                        [0, -2.5, -60],
                                                        [-40, 9.5, -60]], dtype=float).astype(float)  # initial joint position in degrees
    __init_pos_dirction = np.array([[-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [1, -1, -1],
                                                            [1, -1, -1],
                                                            [1, -1, -1]])  
    __init_pos_deg = __init_pos_deg * __init_pos_dirction # adjust the initial position direction
    __init_pos_rad = np.deg2rad(__init_pos_deg)  # initial joint position in radians
    __initjoint_position = __init_pos_rad

    num_body_pos = 3
    num_body_orientation = 3
    num_joint_angles = 18 
    num_forces = 6
    num_foot_traj = 6
    num_contact = 0
    observation_space = np.zeros((num_body_pos + num_body_orientation + num_joint_angles + num_forces + num_foot_traj + num_contact, ), dtype=float).astype(float) 
    action_space = np.zeros((18, ), dtype=float).astype(float)  # joint angles

    #----------- Symmetrical Max-Min Action Range -----------#
    # observation_space_high = np.array([
    #                     7.0512481, -0.013637958,
    #                     0.34497491, 0.35881358, 0.13376351, 0.0721476,
    #                     -0.54595870,  0.76064470, -0.60858520, 0.68776690,  0.43705025, -0.64850265,
    #                     1.26822540,  1.29237580, -0.70805120, 1.38210810, -0.06344602,  2.46140220,
    #                     0.98390025,  0.03390659,  2.27718070, -0.29464778, -0.14714211,  2.50980450,
    #                     11.376931, 23.541754, 18.792133, 10.039366, 19.01429, 18.701794,
    #                     0.42798841, 0.14256591, 0.26136336, 0.18555231, 0.11940541, 0.29765788,
    #                     # 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    #                     ])

    # observation_space_low = np.array([
    #                     -1.5580437, -3.7254312,
    #                     0.17508288, -0.13590206, -0.3235115, -0.63263106, 
    #                     -1.38210810,  0.06344602, -2.46140220, -0.98390025, -0.03390659, -2.27718070,
    #                     0.29464778,  0.14714211, -2.50980450, 0.54595870, -0.76064470,  0.60858520,
    #                     -0.68776690, -0.43705025,  0.64850265, -1.26822540, -1.29237580,  0.70805120,
    #                     0., 0., 0., 0., 0., 0.,
    #                     -0.06665716, 0.00653887, 0.00611944, 0.0062973, 0.00600731, 0.00665024,
    #                     # 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    #                     ])
    
    # action_space_high = np.array([
    #                     -0.08928384,  0.64018328,  0.73880163,
    #                     0.71728384,  0.48140574,  0.52528891,
    #                     0.71734355,  0.82676823,  0.41873297,
    #                     0.87071587,  0.22123349,  1.43173578,
    #                     0.90740824,  0.03776942,  1.41971448,
    #                     0.44833541, -0.02909280,  1.56461933
    #                     ])

    # action_space_low = np.array([
    #                     -0.87071587, -0.22123349, -1.43173578,
    #                     -0.90740824, -0.03776942, -1.41971448,
    #                     -0.44833541,  0.02909280, -1.56461933,
    #                     0.08928384, -0.64018328, -0.73880163,
    #                     -0.71728384, -0.48140574, -0.52528891,
    #                     -0.71734355, -0.82676823, -0.41873297
    #                     ])

    #----------- Original Max-Min Action Range -----------#
    observation_space_high = np.array([
                        7.0512481, -0.013637958,
                        0.34497491, 0.35881358, 0.13376351, 0.0721476,
                        -0.59375274, 0.68826747, -0.72950184, 0.6576674, 0.42351845, -1.0316935, 
                        1.1950908, 1.2923758, -0.92522711, 1.212834, -0.079413719, 2.4614022,
                        0.98390025, -0.01020806, 2.2536771, -0.29464778, -0.14714211, 2.3562224, 
                        11.376931, 23.541754, 18.792133, 10.039366, 19.01429, 18.701794,
                        0.42798841, 0.14256591, 0.26136336, 0.18555231, 0.11940541, 0.29765788
                        ])
    
    observation_space_low = np.array([
                        -1.5580437, -3.7254312,
                        0.17508288, -0.13590206, -0.3235115, -0.63263106, 
                        -1.3821081, 0.06344602, -2.3497694, -0.7023459, -0.03390659, -2.2771807,
                        0.3254354, 0.27022228, -2.5098045, 0.5459587, -0.7606447, 0.6085852,
                        -0.6877669, -0.43705025, 0.64850265, -1.2682254, -1.1273408, 0.7080512,
                        0., 0., 0., 0., 0., 0.,
                        -0.06665716, 0.00653887, 0.00611944, 0.0062973, 0.00600731, 0.00665024
                        ])
    
    action_space_high = np.array([
                            -0.11296894, 0.57256915, 0.48420778, 
                            0.7126509, 0.46846504, 0.0727683,
                            0.51731489, 0.82676823, 0.25294265, 
                            0.81149222, 0.22123349, 1.43173578,
                            0.90740824, -0.02109119, 1.29287913, 
                            0.42693611, -0.03764155, 1.42620209
                        ]) 
    
    action_space_low = np.array([
                            -0.87071587, -0.19994925, -1.33353355, 
                            -0.75652057, -0.03776942, -1.41971448,
                            -0.44833541, 0.0290928, -1.56461933, 
                            0.08928384, -0.64018328, -0.73880163,
                            -0.71728384, -0.48140574, -0.52528891, 
                            -0.71734355, -0.76640703, -0.41873297
                        ])
    
    def __init__(self, port=23000, OnTimeStep=True, simulation = True):
        if simulation:
            self.client = RemoteAPIClient('localhost', port=port)
            self.sim = self.client.require('sim')
            self.OnTimeStep = OnTimeStep  # Set to True for stepping mode, False for continuous mode
            print('Ontime :', self.OnTimeStep)
            self.sim.setStepping(self.OnTimeStep)  # Enable stepping mode for the simulation

            # joint handle
            for leg in range(self.__joint_handle.shape[0]):
                for joint in range(self.__joint_handle.shape[1]):
                    # print(f'Getting joint handle for {self.__joint_names[joint]}{self.__leg_names[leg % 6]}')
                    self.__joint_handle[leg, joint] = self.sim.getObject(
                        self.__joint_names[joint] + self.__leg_names[leg % 6]
                    )
            self.IMU_robot = self.sim.getObject(self.__IMU_names[0])
            self.IMU_ref = self.sim.getObject(self.__IMU_names[1])
            self.set_robot_joint(np.zeros((18, 1)))
            self.update()

            self.reset()
            print("INFO: VrepInterfaze is initialized successfully.")


        # normalization parameters
        # self._obs_mid = (self.observation_space_high + self.observation_space_low) / 2.0
        # self._obs_scale = (self.observation_space_high - self.observation_space_low) / 2.0
        # print(f"Observation space mid: {self._obs_mid}, scale: {self._obs_scale}")
        self._action_mid = (self.action_space_high + self.action_space_low) / 2.0
        self._action_scale = (self.action_space_high - self.action_space_low) / 2.0
        print(f"Action space mid: {self._action_mid}, scale: {self._action_scale}")


    # ------------------- Normalization -------------------
    def normalize_observation(self, observation):
        observation = np.atleast_2d(observation) # shape (B, obs_dim)
        normalized = []

        # z height
        start = 0
        end = self.num_body_pos
        robot_z = observation[:, start:end]
        robot_z_min = self.observation_space_low[start:end]
        robot_z_max = self.observation_space_high[start:end]
        norm_robot_z = 2 * (robot_z - robot_z_min) / (robot_z_max - robot_z_min) - 1
        normalized.append(norm_robot_z)

        # roll, pitch, yaw: use the unified standard for normalization
        start = self.num_body_pos
        end = self.num_body_pos + self.num_body_orientation
        orientation = observation[:, start:end]
        orientation_low = min(self.observation_space_low[start:end])
        orientation_high = max(self.observation_space_high[start:end])
        norm_orientation = 2 * (orientation - orientation_low) / (orientation_high - orientation_low) - 1
        normalized.append(norm_orientation)

        # joint angles: use seperate normalization for each joint
        start = self.num_body_pos + self.num_body_orientation
        end = self.num_body_pos + self.num_body_orientation + self.num_joint_angles
        joint_angles = observation[:, start:end]
        joint_angles_low = self.observation_space_low[start:end]
        joint_angles_high = self.observation_space_high[start:end]
        norm_joint_angles = 2 * (joint_angles - joint_angles_low) / (joint_angles_high - joint_angles_low) - 1
        normalized.append(norm_joint_angles)

        # forces: use the unified standard for normalization
        start = self.num_body_pos + self.num_body_orientation + self.num_joint_angles
        end = self.num_body_pos + self.num_body_orientation + self.num_joint_angles + self.num_forces
        forces = observation[:, start:end]
        forces_low = min(self.observation_space_low[start:end])
        forces_high = max(self.observation_space_high[start:end])
        norm_forces = 2 * (forces - forces_low) / (forces_high - forces_low) - 1
        normalized.append(norm_forces)

        # foot trajectory: use the unified standard for normalization
        start = self.num_body_pos + self.num_body_orientation + self.num_joint_angles + self.num_forces
        end = self.num_body_pos + self.num_body_orientation + self.num_joint_angles + self.num_forces + self.num_foot_traj
        foot_traj = observation[:, start:end]
        foot_traj_low = min(self.observation_space_low[start:end])
        foot_traj_high = max(self.observation_space_high[start:end])
        norm_foot_traj = 2 * (foot_traj - foot_traj_low) / (foot_traj_high - foot_traj_low) - 1
        normalized.append(norm_foot_traj)

        # # contact: already 0 or 1, no need to normalize
        # start = self.num_body_pos + self.num_body_orientation + self.num_joint_angles + self.num_forces + self.num_foot_traj
        # end = self.num_body_pos + self.num_body_orientation + self.num_joint_angles + self.num_forces + self.num_foot_traj + self.num_contact
        # contact = observation[:, start:end]
        # norm_contact = contact * 2 - 1 # 0 -> -1, 1 -> 1
        # normalized.append(norm_contact)
        
        # check if the observation is a single sample or a batch
        normalized_obs = np.concatenate(normalized, axis=1)
        if observation.shape[0] == 1:
            return normalized_obs[0]
        return normalized_obs
    
    # def normalize_observation(self, observation):
    #     return (observation - self._obs_mid) / self._obs_scale

    def denormalize_observation(self, normalized_obs):
        normalized_obs = np.atleast_2d(normalized_obs)  # shape (B, obs_dim)
        denormalized = []

        # z height
        start = 0
        end = self.num_body_pos
        norm_robot_z = normalized_obs[:, start:end]
        robot_z_min = self.observation_space_low[start:end]
        robot_z_max = self.observation_space_high[start:end]
        robot_z = (norm_robot_z + 1) / 2 * (robot_z_max - robot_z_min) + robot_z_min
        denormalized.append(robot_z)

        # roll, pitch, yaw
        start = end
        end = start + self.num_body_orientation
        norm_orientation = normalized_obs[:, start:end]
        orientation_low = min(self.observation_space_low[start:end])
        orientation_high = max(self.observation_space_high[start:end])
        orientation = (norm_orientation + 1) / 2 * (orientation_high - orientation_low) + orientation_low
        denormalized.append(orientation)

        # joint angles
        start = end
        end = start + self.num_joint_angles
        norm_joint_angles = normalized_obs[:, start:end]
        joint_angles_low = self.observation_space_low[start:end]
        joint_angles_high = self.observation_space_high[start:end]
        joint_angles = (norm_joint_angles + 1) / 2 * (joint_angles_high - joint_angles_low) + joint_angles_low
        denormalized.append(joint_angles)

        # forces
        start = end
        end = start + self.num_forces
        norm_forces = normalized_obs[:, start:end]
        forces_low = min(self.observation_space_low[start:end])
        forces_high = max(self.observation_space_high[start:end])
        forces = (norm_forces + 1) / 2 * (forces_high - forces_low) + forces_low
        denormalized.append(forces)

        # foot trajectory
        start = end
        end = start + self.num_foot_traj
        norm_foot_traj = normalized_obs[:, start:end]
        foot_traj_low = min(self.observation_space_low[start:end])
        foot_traj_high = max(self.observation_space_high[start:end])
        foot_traj = (norm_foot_traj + 1) / 2 * (foot_traj_high - foot_traj_low) + foot_traj_low
        denormalized.append(foot_traj)

        # # contact: if you use the [-1, 1] mapping, you can invert like this:
        # start = end
        # end = start + self.num_contact
        # norm_contact = normalized_obs[:, start:end]
        # contact = (norm_contact + 1) / 2  # -1 -> 0, 1 -> 1
        # denormalized.append(contact)

        # check if the observation is a single sample or a batch
        denormalized_obs = np.concatenate(denormalized, axis=1)
        if normalized_obs.shape[0] == 1:
            return denormalized_obs[0]
        return denormalized_obs

    def normalize_action(self, action):
        return (action - self._action_mid) / self._action_scale

    def denormalize_action(self, norm_action):
        return norm_action * self._action_scale + self._action_mid

    def normalize_expert_data(self, expert_data):
        expert_data['state'] = self.normalize_observation(expert_data['state'])
        expert_data['action'] = self.normalize_action(expert_data['action'])
        return expert_data


    # ---------------------- actuation ------------------------
    def set_robot_joint(self, target_pos):
        target_pos = target_pos.reshape((6, 3))
        offset = self.__initjoint_position
        for leg in range(0, 6):
            target_pos[leg] += offset[leg]
        self.__target_positions = target_pos

    def set_zero(self):
        self.set_robot_joint(np.zeros(18))


    # ---------------------- get simulation data ------------------------
    def get_jointangle(self):
        positions = np.zeros((18))
        for l in range(self.__joint_handle.shape[0]):
            for j in range(self.__joint_handle.shape[1]):
                positions[3 * l + j] = self.sim.getJointPosition(int(self.__joint_handle[l][j]))
        return positions
    
    def get_bodyposition(self):
        robot_pos = np.zeros((3))
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        robot_z = robot_pos[2]
        robot_z = np.array([robot_z]).reshape((1,))
        return robot_pos

    def get_bodyorientation(self):
        orientation = np.zeros((3))
        orientation = self.sim.getObjectOrientation(self.IMU_robot, self.IMU_ref)
        return orientation
    
    def get_force(self):
        forces = np.zeros((6))
        for i in range(6):
            _, forceVector, _ = self.sim.readForceSensor(self.sim.getObject(self.__forcesensor_names[i]))
            forces[i] = max(0, np.sqrt((forceVector[0])**2 + (forceVector[1])**2 + (forceVector[2])**2) - 0.2)
        return forces
    
    def get_foot_trajectory(self):
        foot_traj = np.zeros((6))
        for i in range(6):
            # Get the foor trajectory z
            foot_traj[i] = self.sim.getObjectPosition(self.sim.getObject(self.__foot_names[i]))[2]
        return foot_traj
    
    def get_contact(self):
        # data[contact_col] = (data[force_col].abs() > threshold).astype(int)
        contact = np.zeros((6))
        forces = self.get_force()
        for i in range(6):
            contact[i] = 1 if forces[i] > 0.5 else 0
        return contact
    
    def get_states(self):
        body_pos = self.get_bodyposition()
        body_orientation = self.get_bodyorientation()
        joint_angles = self.get_jointangle()
        forces = self.get_force()
        foot_traj = self.get_foot_trajectory()
        contact = self.get_contact()
        states = np.concatenate((body_pos, body_orientation, joint_angles, forces, foot_traj)) #, forces, foot_traj, contact
        return states


    # ---------------------- simulation control ------------------------
    def update(self):
        for leg in range(self.__joint_handle.shape[0]):
            for joint in range(self.__joint_handle.shape[1]):
                self.sim.setJointTargetPosition(int(self.__joint_handle[leg][joint]),
                                                self.__target_positions[leg][joint])
        if self.OnTimeStep:
            self.sim.step()

    def reset(self, zero=True):
        self.stop()
        time.sleep(1)
        self.start()
        # record the initial position of the robot
        head_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        self._previous_x = head_pos[0]
        # reset the robot joints to zero or initial position
        if zero:
            self.set_zero()
            noise = np.random.uniform(-0.1, 0.1, size=(18, ))
            self.set_robot_joint(noise)
            self.update()
        # add noise to the initial states 
        # the reset state will be sent to the actor networks: need to normalize it first
        obs = self.get_states()
        obs = self.normalize_observation(obs)
        noise_obs = obs + np.random.normal(-0.1, 0.1, size=obs.shape)
        # reset the step count
        self._step_count = 0
        return noise_obs
    
    def is_healthy(self):
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        robot_height = robot_pos[2]
        return robot_height > 0.1

    def step(self,action):
        # recieive the policy action and denormalize it
        action = self.denormalize_action(action)
        self.set_robot_joint(action)
        self.update() 
        obs = self.get_states()
        # normalize the observation for the policy
        obs = self.normalize_observation(obs)
        # clip the observation to the observation space bounds
        # obs = np.clip(obs, 0.0, 1.0)

        # calculate the reward based on the robot's position
        reward = self.sim.getObjectVelocity(self.sim.getObject('/head'))[0][0] * 100

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        terminated = False
        # terminated = not self.is_healthy() 

        return obs, reward, terminated, truncated, {}  

    def start(self):
        self.sim.setStepping(self.OnTimeStep)
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()


if __name__ == "__main__":
    env = CoppeliaSimEnv()
    env.reset()
    # env.start()
    for i in range(100):
        action = np.random.uniform(-1, 1, size=18)
        obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.get_states()
        # print(f"Step {i+1}, Action: {action}, States: {next_obs}")
    env.stop()
