import zmq
import msgpack
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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

    observation_space = np.zeros((3 + 18 + 6 + 6, ), dtype=float).astype(float)  # body orientation, joint angles, forces, foot trajectory
    action_space = np.zeros((18, ), dtype=float).astype(float)  # joint angles

    observation_space_high = np.array([
                        0.26309153, 0.14352329, 0.11150103, 0.20889758, -0.5805428, 0.917541,
                        -0.46537167, 0.71234864, 0.6082702, -0.5482097, 1.1905043, 0.9968296,
                        -0.7116102, 1.1811888, -0.03991236, 2.3925962, 0.90482897, 0.1928462,
                        2.266345, -0.27362847, -0.13811275, 2.4670823, 6.9036007, 14.140027,
                        13.357503, 11.274531, 15.725126, 9.445448, 0.69686425, 0.19966786,
                        0.11533475, 0.68759626, 0.24561436, 0.1978422
                        ])
    
    observation_space_low = np.array([
                        0.19126146, -0.03429405, -0.16010809, -0.08192597, -1.4591902, 0.01655765,
                        -2.3673697, -0.7474498, -0.2489663, -2.2368977, 0.27859572, 0.20474741,
                        -2.5910378, 0.59677696, -0.91530794, 0.32074368, -0.54582196, -0.50561714,
                        0.35649562, -1.2296814, -0.9988971, 0.48642993, 0., 0.,
                        0., 0., 0., 0., 0.00760507, 0.00926453,
                        0.00918054, 0.00917197, 0.00868974, 0.00918513
                        ])

    action_space_high = np.array([
                            -0.57568526, 0.91962415, -0.4638236, 0.71432644, 0.6131103, -0.5386828,
                            1.1927193, 1.0053458, -0.7053654, 1.182008, -0.03557599, 2.3935604,
                            0.9115751, 0.20103599, 2.268618, -0.27113712, -0.13196065, 2.4745364
                        ]) 

    action_space_low = np.array([
                            -1.4617796, -0.01609878, -2.369329, -0.75536764, -0.2783127, -2.2406516,
                            0.27379432,  0.17258137, -2.5969138,  0.5853528, -0.9162319,  0.31624538,
                            -0.54989666, -0.5026367,  0.34887356, -1.2245036, -1.002201,  0.48189536
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
        self._obs_mid = (self.observation_space_high + self.observation_space_low) / 2.0
        self._obs_scale = (self.observation_space_high - self.observation_space_low) / 2.0
        print(f"Observation space mid: {self._obs_mid}, scale: {self._obs_scale}")
        self._action_mid = (self.action_space_high + self.action_space_low) / 2.0
        self._action_scale = (self.action_space_high - self.action_space_low) / 2.0
        print(f"Action space mid: {self._action_mid}, scale: {self._action_scale}")

        self.reset()
        print("INFO: VrepInterfaze is initialized successfully.")


    # ------------------- Normalization -------------------
    def normalize_observation(self, observation):
        return (observation - self._obs_mid) / self._obs_scale
    
    def denormalize_observation(self, norm_observation):
        return norm_observation * self._obs_scale + self._obs_mid
    
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
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        robot_z = robot_pos[2]
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
        body_z = self.get_bodyposition()
        body_orientation = self.get_bodyorientation()
        joint_angles = self.get_jointangle()
        forces = self.get_force()
        foot_traj = self.get_foot_trajectory()
        states = np.concatenate((body_z, body_orientation, joint_angles, forces, foot_traj))
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

        # calculate the reward based on the robot's position
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        current_x = robot_pos[0]
        reward = current_x * 10
        self._previous_x = current_x

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        terminated = not self.is_healthy() 

        return obs, reward, terminated, truncated, {}  

    def start(self):
        self.sim.setStepping(self.OnTimeStep)
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()


if __name__ == "__main__":
    env = CoppeliaSimEnv()
    env.reset()
    env.start()
    for i in range(100):
        action = np.random.uniform(-1, 1, size=18)
        obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.get_states()
        print(f"Step {i+1}, Action: {action}, States: {next_obs}")
    env.stop()
