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
    __initjoint_position = np.zeros((18, 1), dtype=float).astype(float)

    observation_space = np.zeros((1 + 3 + 18 + 6 + 6, ), dtype=float).astype(float)  # body orientation, joint angles, forces, foot trajectory
    action_space = np.zeros((18, ), dtype=float).astype(float)  # joint angles

    observation_space_high = np.array([
                        0.34497491, 0.35881358, 0.13376351, 0.0721476,
                        -0.59375274, 0.68826747, -0.72950184, 0.6576674, 0.42351845, -1.0316935, 
                        1.1950908, 1.2923758, -0.92522711, 1.212834, -0.079413719, 2.4614022,
                        0.98390025, -0.01020806, 2.2536771, -0.29464778, -0.14714211, 2.3562224, 
                        11.376931, 23.541754, 18.792133, 10.039366, 19.01429, 18.701794,
                        0.42798841, 0.14256591, 0.26136336, 0.18555231, 0.11940541, 0.29765788
                        ])
    
    observation_space_low = np.array([
                        0.17508288, -0.13590206, -0.3235115, -0.63263106, 
                        -1.3821081, 0.06344602, -2.3497694, -0.7023459, -0.03390659, -2.2771807,
                        0.3254354, 0.27022228, -2.5098045, 0.5459587, -0.7606447, 0.6085852,
                        -0.6877669, -0.43705025, 0.64850265, -1.2682254, -1.1273408, 0.7080512,
                        0., 0., 0., 0., 0., 0.,
                        -0.06665716, 0.00653887, 0.00611944, 0.0062973, 0.00600731, 0.00665024
                        ])

    action_space_high = np.array([
                            -0.6365677, 0.7383754, -0.5629898, 0.7126509, 0.4248318, -0.97442925,
                            1.2154466, 0.9925745, -0.7942549, 1.335091, 0.05542721, 2.4789333,
                            0.90740824, 0.02254204, 2.3400767, -0.2711956, -0.20344783, 2.4733996
                        ]) 

    action_space_low = np.array([
                            -1.3943146, -0.03414297, -2.380731, -0.75652057, -0.08140265, -2.466912,
                            0.24979629, 0.19489908, -2.611817, 0.6128826, -0.80598956, 0.30839592,
                            -0.71728384, -0.4377725, 0.52190864, -1.4154752, -0.9322133, 0.6284646
                        ])

    def __init__(self, port=23000, OnTimeStep=True):
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
        self.__initjoint_position = self.get_jointangle()
        self.set_robot_joint(np.zeros((18, 1)))
        self.update()

        # normalization parameters
        # self._obs_mid = (self.observation_space_high + self.observation_space_low) / 2.0
        # self._obs_scale = (self.observation_space_high - self.observation_space_low) / 2.0
        # print(f"Observation space mid: {self._obs_mid}, scale: {self._obs_scale}")
        self._action_mid = (self.action_space_high + self.action_space_low) / 2.0
        self._action_scale = (self.action_space_high - self.action_space_low) / 2.0
        print(f"Action space mid: {self._action_mid}, scale: {self._action_scale}")

        self.reset()
        print("INFO: VrepInterfaze is initialized successfully.")


    # ------------------- Normalization -------------------
    def normalize_observation(self, observation):
        observation = np.atleast_2d(observation) # shape (B, obs_dim)
        normalized = []

        # z height
        robot_z = observation[:, 0:1]
        robot_z_min = self.observation_space_low[0]
        robot_z_max = self.observation_space_high[0]
        norm_robot_z = 2 * (robot_z - robot_z_min) / (robot_z_max - robot_z_min) - 1
        normalized.append(norm_robot_z)

        # roll, pitch, yaw: use the unified standard for normalization
        orientation = observation[:, 1:4]
        orientation_low = min(self.observation_space_low[1:4])
        orientation_high = max(self.observation_space_high[1:4])
        norm_orientation = 2 * (orientation - orientation_low) / (orientation_high - orientation_low) - 1
        normalized.append(norm_orientation)

        # joint angles: use seperate normalization for each joint
        joint_angles = observation[:, 4:22]
        joint_angles_low = self.observation_space_low[4:22]
        joint_angles_high = self.observation_space_high[4:22]
        norm_joint_angles = 2 * (joint_angles - joint_angles_low) / (joint_angles_high - joint_angles_low) - 1
        normalized.append(norm_joint_angles)
        # norm_joint_angles = np.zeros_like(joint_angles)
        # for i in range(len(joint_angles)):
        #     norm_joint_angles[i] = (joint_angles[i] - joint_angles_low[i]) / (joint_angles_high[i] - joint_angles_low[i])

        # forces: use the unified standard for normalization
        forces = observation[:, 22:28]
        forces_low = min(self.observation_space_low[22:28])
        forces_high = max(self.observation_space_high[22:28])
        norm_forces = 2 * (forces - forces_low) / (forces_high - forces_low) - 1
        normalized.append(norm_forces)

        # foot trajectory: use the unified standard for normalization
        foot_traj = observation[:, 28:34]
        foot_traj_low = min(self.observation_space_low[28:34])
        foot_traj_high = max(self.observation_space_high[28:34])
        norm_foot_traj = 2 * (foot_traj - foot_traj_low) / (foot_traj_high - foot_traj_low) - 1
        normalized.append(norm_foot_traj)
        
        # check if the observation is a single sample or a batch
        normalized_obs = np.concatenate(normalized, axis=1)
        if observation.shape[0] == 1:
            return normalized_obs[0]
        return normalized_obs
    
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
        offset = self.__initjoint_position.reshape((6, 3))
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
        return robot_z

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
    
    def get_states(self):
        body_pos = self.get_bodyposition()
        body_orientation = self.get_bodyorientation()
        joint_angles = self.get_jointangle()
        forces = self.get_force()
        foot_traj = self.get_foot_trajectory()
        states = np.concatenate((body_pos, body_orientation, joint_angles, forces, foot_traj))
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
        # reset the step count
        self._step_count = 0
        return self.get_states()
    
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
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        current_x = robot_pos[0]
        reward = current_x * 10
        self._previous_x = current_x

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
