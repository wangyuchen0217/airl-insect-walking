# interface/VREPZMQ.py

'''
Class: VREP_Interface

This class provide easy direct interface between python3 and CoppeliaSim
aiming mainly for reinforcement learning

NOTE THAT: this class use numpy array
'''

# ------------------- import modules ---------------------

# ZMQ Remote API for communication with VREP
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# math-relate modules
import numpy as np
import time


# ------------------- configuration variables ---------------------
class VREP_Interface():

    __joint_name = ['/T', '/C', '/F']  # predefined joint names
    __graph_name = ['ForceL1,ForceL2,ForceL3,ForceR1,ForceR2,ForceR3']

    __forcesensor_handle = np.zeros((6, 1), dtype=int).astype(int)
    __joint_handle = np.zeros((6, 3), dtype=int).astype(int)  # joint handle (leg l, joint j)
    __target_positions = np.zeros((6, 3), dtype=float).astype(float)  # joint target position (leg l, joint j)
    __initjoint_position = np.zeros((18, 1), dtype=float).astype(float)
    __foot_handle = np.zeros((6, 1), dtype=int).astype(int)
    __graph_handle = np.zeros((1, 3), dtype=int).astype(int)

    # ---------------------- constructor ------------------------
    def __init__(self, port=23000, OnTimeStep=True):
        self.client = RemoteAPIClient('localhost', port=port)
        self.sim = self.client.require('sim')
        self.OnTimeStep = OnTimeStep
        print('Ontime :', OnTimeStep)
        self.sim.setStepping(self.OnTimeStep)

        # robot handle
        self.__robot_ref_handle = self.sim.getObject('/Ref_Frame')

        # joint handle
        for leg in range(self.__joint_handle.shape[0]):
            for joint in range(self.__joint_handle.shape[1]):
                self.__joint_handle[leg, joint] = self.sim.getObject(
                    self.__joint_name[joint] + ('L' if leg > 2 else 'R') + str(leg % 3)
                )
        self.__initjoint_position = self.get_jointangle()
        self.set_robot_joint(np.zeros((18, 1)))
        self.update()

        # forcesensor handle and foottip handle
        for leg in range(self.__forcesensor_handle.shape[0]):
            self.__forcesensor_handle[leg] = self.sim.getObject('/' + ('L' if leg > 2 else 'R') + str(leg % 3) + '_fs')
            self.__foot_handle[leg] = self.sim.getObject('/' + ('FOOT_L' if leg > 2 else 'FOOT_R') + str(leg % 3))

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
    def get_robot_pose(self, relativeworld=False):
        posarray = self.sim.getObjectPosition(int(self.__robot_ref_handle))
        orienarray = self.sim.getObjectOrientation(int(self.__robot_ref_handle))
        pose = np.array([*posarray, *orienarray])

        linear, angular = self.sim.getObjectVelocity(int(self.__robot_ref_handle))
        robotframe = self.sim.getObjectMatrix(int(self.__robot_ref_handle), self.sim.handle_world)
        newrobotframe = self.sim.getMatrixInverse(robotframe)
        newrobotframe[3] = 0
        newrobotframe[7] = 0
        newrobotframe[11] = 0
        linearVelo = self.sim.multiplyVector(newrobotframe, linear)
        angularVelo = self.sim.multiplyVector(newrobotframe, angular)

        if relativeworld:
            velocity = np.array([*linear, *angular])
        else:
            velocity = np.array([*linearVelo, *angularVelo])

        return pose, velocity

    def get_joint_focejoint(self):
        joint_force = np.zeros((6, 3))
        for l in range(0, 6):
            for j in range(0, 3):
                joint_force[l][j] = self.sim.getJointForce(int(self.__joint_handle[l][j]))
        return joint_force

    def get_joint_position(self):
        joint_position = np.zeros((6, 3))
        for l in range(0, 6):
            for j in range(0, 3):
                joint_position[l][j] = self.sim.getJointPosition(int(self.__joint_handle[l][j]))
        return joint_position

    def get_joint_velocity(self):
        joint_velocity = np.zeros((6, 3))
        for l in range(0, 6):
            for j in range(0, 3):
                joint_velocity[l][j] = self.sim.getJointVelocity(int(self.__joint_handle[l][j]))
        return joint_velocity

    def get_footvelocity(self):
        footvelocity = np.zeros((6, 1))
        for leg in range(0, 6):
            linearVelocity, _ = self.sim.getObjectVelocity(int(self.__forcesensor_handle[leg]))
            footvelocity[leg] = abs(linearVelocity[0]) + abs(linearVelocity[1])
        return footvelocity

    def get_footforce(self):
        forces = np.zeros((6, 1))
        forceflag = np.zeros((6, 1))
        force_all = np.zeros((6, 3))
        for leg in range(0, 6):
            result, forceVector, _ = self.sim.readForceSensor(int(self.__forcesensor_handle[leg]))
            forces[leg] = max(forceVector[2] * -1, 0)
            forceflag[leg] = result
            force_all[leg] = forceVector
        return forces, forceflag, force_all

    def get_footposition(self):
        footposition = np.zeros((6, 1))
        for leg in range(0, 6):
            _, _, z = self.sim.getObjectPosition(int(self.__forcesensor_handle[leg]))
            footposition[leg] = z
        return footposition

    def get_jointangle(self):
        positions = np.zeros((18))
        for l in range(0, 6):
            for j in range(0, 3):
                positions[3 * l + j] = self.sim.getJointPosition(int(self.__joint_handle[l][j]))
        return positions

    # ---------------------- simulation control ------------------------
    def update(self):
        for leg in range(0, 6):
            for joint in range(0, 3):
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

    def start(self):
        self.sim.setStepping(self.OnTimeStep)
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()


# ---------------------- main for test ------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    interface = VREP_Interface(OnTimeStep=True, port=9999)
    time.sleep(0.1)
    interface.start()
    print('::Check Velocity and Force Reward::')

    plt.style.use('seaborn')
    fig, ax = plt.subplots(4, 1, figsize=(10, 12))

    time_data = []
    velocity_data = np.zeros((6, 0))
    foot_velocity_data = np.zeros((6, 0))
    foot_force_data = np.zeros((6, 0))

    linear_lines = [ax[0].plot([], [], label=label)[0] for label in ['vx', 'vy', 'vz']]
    angular_lines = [ax[1].plot([], [], label=label)[0] for label in ['wx', 'wy', 'wz']]
    foot_velocity_lines = [ax[2].plot([], [], label=f'Foot {i + 1}')[0] for i in range(6)]
    foot_force_lines = [ax[3].plot([], [], label=f'Foot {i + 1}')[0] for i in range(6)]

    for axis, title in zip(ax, ['Linear Velocity', 'Angular Velocity', 'Foot Velocity', 'Foot Force']):
        axis.set_xlim(0, 10)
        axis.set_ylim(-1, 1)
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Value')
        axis.legend(loc='upper right')
        axis.set_title(title)

    ax[3].set_ylim(0, 50)

    def update(frame):
        global velocity_data, foot_velocity_data, foot_force_data
        interface.update()

        velocity = interface.get_robot_pose()[1]
        foot_velocity = interface.get_footvelocity()
        foot_force, _, _ = interface.get_footforce()

        time_data.append(frame * 0.1)
        velocity_data = np.hstack((velocity_data, velocity.reshape(-1, 1)))
        foot_velocity_data = np.hstack((foot_velocity_data, foot_velocity))
        foot_force_data = np.hstack((foot_force_data, foot_force))

        for i, line in enumerate(linear_lines):
            line.set_data(time_data, velocity_data[i])

        for i, line in enumerate(angular_lines):
            line.set_data(time_data, velocity_data[i + 3])

        for i, line in enumerate(foot_velocity_lines):
            line.set_data(time_data, foot_velocity_data[i])

        for i, line in enumerate(foot_force_lines):
            line.set_data(time_data, foot_force_data[i])

    ani = FuncAnimation(fig, update, interval=100)
    plt.tight_layout()
    plt.show()
