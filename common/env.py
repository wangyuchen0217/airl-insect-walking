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

        self.reset()
        print("INFO: VrepInterfaze is initialized successfully.")


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
        body_orientation = self.get_bodyorientation()
        joint_angles = self.get_jointangle()
        forces = self.get_force()
        foot_traj = self.get_foot_trajectory()
        states = np.concatenate((body_orientation, joint_angles, forces, foot_traj))
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
        if zero:
            self.set_zero()
            self.update()
        return self.get_states()
    
    def is_healthy(self):
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        robot_height = robot_pos[2]
        return robot_height > 0.1

    def step(self,action):
        self.set_robot_joint(action)
        self.update() 
        obs = self.get_states()
        reward = 0.0

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
        action = np.random.uniform(-0.5, 0.5, size=18)
        obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.get_states()
        print(f"Step {i+1}, Action: {action}, States: {next_obs}")
    env.stop()
