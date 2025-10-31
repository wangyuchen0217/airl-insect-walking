import zmq
import msgpack
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class ObsField:
    name: str               # The name of this field
    size: int               # The dimension of this field
    getter: str             # Get the function name in the environment class
    norm: str               # Normalization method: 'unified', 'separate', 'none'
    low: np.ndarray | float  #  The lower bound of this field (can be scalar or np array; binary can be None)
    high: np.ndarray | float  # The upper bound of this field (can be scalar or np array; binary can be None)
    include: bool = True    # whether to include this field in the observation

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
    __target_positions = np.zeros((5, 3), dtype=float).astype(float)  # joint target position (leg l, joint j)
    __initjoint_position = np.zeros((5, 3), dtype=float).astype(float)  # initial joint position (leg l, joint j)
    __init_pos_deg = np.array([[30, 9.5, -60], 
                                                        [ 0 ,  -2.5, -60],
                                                        [-40, 9.5,-60],
                                                        [30, 9.5, -60], 
                                                        # [0, -2.5, -60],
                                                        [-40, 9.5, -60]], dtype=float).astype(float)  # initial joint position in degrees
    __init_pos_dirction = np.array([[-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [1, -1, -1],
                                                            # [1, -1, -1],
                                                            [1, -1, -1]])  
    __init_pos_deg = __init_pos_deg * __init_pos_dirction # adjust the initial position direction
    __init_pos_rad = np.deg2rad(__init_pos_deg)  # initial joint position in radians
    __initjoint_position = __init_pos_rad

    OBS_SPEC: tuple[ObsField, ...] = (

                    ObsField('body_pos',      1,  'get_bodyposition',   'per_dim',
                                low=np.array([0.19342157]),  # np.array([-1.5743479, -0.13103247, 0.19342157]), 
                                high=np.array([0.2718982]),  # np.array([-0.01882718, 0.49992156, 0.2718982]), 
                                include=True), # True, False

                    ObsField('orientation',   3,  'get_bodyorientation','shared',
                                low=min([-0.1253066, -0.21079601, -0.14037536]),  
                                high=max([0.17421827, 0.03616637, 0.56608814]),
                                include=True),

                    # ObsField('orientation',   3,  'get_bodyorientation','shared',
                    #             low=min([-0.5, -0.5, -1.5]),  
                    #             high=max([0.5, 0.5, 1.5]),
                    #             include=True),                                

                    ObsField('joint_angles', 15, 'get_jointangle', 'per_dim',
                                low=np.array([
                                    -1.3860602,  0.06034265, -2.4969175, -0.9650939, -0.0351965 , -2.3240883,  
                                    0.28500196, 0.15376441, -2.507828, 0.5072578, -0.7540891 ,  0.5758628,  
                                    # -0.6832747, -0.6967423 ,  0.64540935, -1.2671623, -1.0010364 ,  0.58814275  
                                    -1.2671623, -1.0010364 ,  0.58814275  
                                ]),
                                high=np.array([
                                    -0.5072578 ,  0.7540891 , -0.5758628, 0.6832747 ,  0.6967423 , -0.64540935, 
                                    1.2671623 ,  1.0010364 , -0.58814275, 1.3860602 , -0.06034265,  2.4969175,  
                                    # 0.9650939 ,  0.0351965 ,  2.3240883, -0.28500196, -0.15376441,  2.507828    
                                    -0.28500196, -0.15376441,  2.507828    
                                ]),
                                include=True
                            ), 

                    ObsField('qvel_body', 6, 'get_qvel_body', 'shared',
                                low=min([-0.13889849, -0.41793427, -0.7163129, -1.5593499, -1.6832889, -1.5487039]),
                                high=max([0.86493146, 0.5213626, 0.32529727, 1.3493888, 1.0914965, 1.5461688]), 
                                include=False),

                    ObsField('qvel_joints', 18, 'get_qvel_joints', 'per_dim',
                            low=np.array([
                                -6.34015036, -5.73144054, -6.28769588, -6.40328217, -3.32265925, -6.28675842,
                                -6.27953339, -3.38050628, -6.29043961, -1.96826971, -2.68654227, -6.29233313,
                                -6.3885498,  -6.2946496,  -6.28978157, -4.32833147, -6.36295366, -4.85788155
                            ]),
                            high=np.array([
                                1.96826971, 2.68654227, 6.29233313, 6.3885498,  6.2946496,  6.28978157,
                                4.32833147, 6.36295366, 4.85788155, 6.34015036, 5.73144054, 6.28769588,
                                6.40328217, 3.32265925, 6.28675842, 6.27953339, 3.38050628, 6.29043961
                            ]),
                            include=False),

                    ObsField('forces',        6,  'get_force',          'shared',
                                low=0.0,
                                high=max([11.871004, 22.840376, 20.059353, 14.028709, 28.488878, 13.580413]),
                                include=False),  

                    ObsField('foot_traj',     6,  'get_foot_trajectory','shared',
                                low=min([0.00600649, 0.00653866, 0.00605668, 0.00703868, 0.00606308, 0.00643684]), 
                                high=max([0.30520386, 0.16491821, 0.10601486, 0.29555720, 0.08111896, 0.12053363]),
                                include=False),   
                            
                    ObsField('contact',       6,  'get_contact',        'binary',
                                low=None, high=None, 
                                include=True))

    action_space_high = np.array([
                        -0.08928384, 0.64018328, 0.73880163, 
                        0.71728384, 0.53050838, 0.52528891,
                        0.61509333, 0.76640703, 0.46057537, 
                        0.87071587, 0.19994925, 1.43173578,
                        # 0.90740824, 0.03776942, 1.30299309, 
                        0.44833541, 0.00427082, 1.59938887
                        ])

    action_space_low = np.array([
                        -0.87071587, -0.19994925, -1.43173578, 
                        -0.90740824, -0.03776942, -1.30299309,
                        -0.44833541, -0.00427082, -1.59938887, 
                        0.08928384, -0.64018328, -0.73880163,
                        # -0.71728384, -0.53050838, -0.52528891, 
                        -0.61509333, -0.76640703, -0.46057537
                        ])
    
    def __init__(self, port=23000, OnTimeStep=True, simulation = True):
        # build observation layout
        self._build_obs_layout()

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
            self.set_robot_joint(np.zeros((15, 1)))
            self.update()

            self.reset()
            print("INFO: VrepInterfaze is initialized successfully.")

        # normalization parameters for action space
        self._action_mid = (self.action_space_high + self.action_space_low) / 2.0
        self._action_scale = (self.action_space_high - self.action_space_low) / 2.0
        # print(f"Action space mid: {self._action_mid}, scale: {self._action_scale}")


    # ------------------- Build obs layout ------------------- #
    def _build_obs_layout(self):
        self.obs_fields = [f for f in self.OBS_SPEC if f.include]
        # build slices and observation space bounds
        idx = 0
        self.slices = {}
        lows, highs = [], []
        for f in self.obs_fields:
            sl = slice(idx, idx + f.size)
            self.slices[f.name] = sl
            idx += f.size

            # expand low/high to array if needed
            low = np.full((f.size,), f.low)  if np.isscalar(f.low)  or f.low is None else np.asarray(f.low).reshape(-1)
            high= np.full((f.size,), f.high) if np.isscalar(f.high) or f.high is None else np.asarray(f.high).reshape(-1)
            # binary features are always in [0, 1]
            if f.norm != 'binary':
                lows.append(low)
                highs.append(high)
            else:
                lows.append(np.zeros((f.size,)))
                highs.append(np.ones((f.size,)))

        self.obs_dim = idx
        self.observation_space_low  = np.concatenate(lows, axis=0).astype(float)
        self.observation_space_high = np.concatenate(highs, axis=0).astype(float)

        self.observation_space = np.zeros((self.obs_dim,), dtype=float)
        self.action_space = np.zeros((self.action_space_low.shape[0],), dtype=float)

    # ------------------- Normalization ------------------- #
    def normalize_observation(self, obs):
        obs = np.atleast_2d(obs).astype(float)  # (B, obs_dim)
        out = np.zeros_like(obs)

        for f in self.obs_fields:
            sl = self.slices[f.name]
            x = obs[:, sl]

            if f.norm == 'per_dim':
                low = self.observation_space_low[sl]
                high = self.observation_space_high[sl]
                out[:, sl] = 2.0 * (x - low) / (high - low) - 1.0

            elif f.norm == 'shared':
                low = float(np.min(self.observation_space_low[sl]))
                high = float(np.max(self.observation_space_high[sl]))
                out[:, sl] = 2.0 * (x - low) / (high - low) - 1.0

            elif f.norm == 'binary':
                out[:, sl] = x * 2.0 - 1.0 # 0 -> -1, 1 -> 1
            else:
                raise ValueError(f"Unknown norm: {f.norm}")
        return out[0] if out.shape[0] == 1 else out
    
    def denormalize_observation(self, norm_obs):
        norm_obs = np.atleast_2d(norm_obs).astype(float)
        out = np.zeros_like(norm_obs)

        for f in self.obs_fields:
            sl = self.slices[f.name]
            xn = norm_obs[:, sl]

            if f.norm == 'per_dim':
                low = self.observation_space_low[sl]
                high = self.observation_space_high[sl]
                out[:, sl] = (xn + 1.0) / 2.0 * (high - low) + low

            elif f.norm == 'shared':
                low = float(np.min(self.observation_space_low[sl]))
                high = float(np.max(self.observation_space_high[sl]))
                out[:, sl] = (xn + 1.0) / 2.0 * (high - low) + low

            elif f.norm == 'binary':
                out[:, sl] = (xn + 1.0) / 2.0 # -1 -> 0, 1 -> 1
            else:
                raise ValueError(f"Unknown norm: {f.norm}")
        return out[0] if out.shape[0] == 1 else out

    def normalize_action(self, action):
        return (action - self._action_mid) / self._action_scale

    def denormalize_action(self, norm_action):
        return norm_action * self._action_scale + self._action_mid

    def normalize_expert_data(self, expert_data):
        expert_data['state'] = self.normalize_observation(expert_data['state'])
        expert_data['action'] = self.normalize_action(expert_data['action'])
        return expert_data


    # ---------------------- actuation ------------------------ #
    def set_robot_joint(self, target_pos):
        target_pos = target_pos.reshape((5, 3))
        offset = self.__initjoint_position
        for leg in range(0, 5):
            target_pos[leg] += offset[leg]
        self.__target_positions = target_pos

    def set_zero(self):
        self.set_robot_joint(np.zeros(15))


    # ---------------------- get simulation data ------------------------ #
    def get_jointangle(self):
        positions = np.zeros((18))
        for l in range(self.__joint_handle.shape[0]):
            for j in range(self.__joint_handle.shape[1]):
                positions[3 * l + j] = self.sim.getJointPosition(int(self.__joint_handle[l][j]))
        # remove the MR data
        positions = np.delete(positions, [12,13,14])
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
    
    def get_qvel_body(self):
        qvel_body = np.zeros((6))
        qvel_body_get = self.sim.getObjectVelocity(self.sim.getObject('/head'))
        qvel_body = np.array(qvel_body_get[0] + qvel_body_get[1]).reshape((6,))
        return qvel_body

    def get_qvel_joints(self):
        qvel_joints = np.zeros((18))
        for l in range(self.__joint_handle.shape[0]):
            for j in range(self.__joint_handle.shape[1]):
                qvel_joints[3 * l + j] = self.sim.getJointVelocity(int(self.__joint_handle[l][j]))
        # remove the MR data
        qvel_joints = np.delete(qvel_joints, [12,13,14])
        return qvel_joints
    
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
        # contact filtered by force sensor
        contact = np.zeros((6))
        forces = self.get_force()
        for i in range(6):
            contact[i] = 1 if forces[i] > 0.27 else 0
        return contact
    
    def get_states(self):
        parts = []
        for f in self.obs_fields:
            # Use the getter method to retrieve the field values
            vals = getattr(self, f.getter)()
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if vals.size != f.size:
                raise RuntimeError(f"{f.name} size mismatch: got {vals.size}, expected {f.size}")
            parts.append(vals)
        return np.concatenate(parts, axis=0)

    # ---------------------- simulation control ------------------------ #
    def update(self):
        # remove RM leg (row 4)
        self.__target_handle = np.delete(self.__joint_handle, 4, axis=0)
        for leg in range(self.__target_handle.shape[0]):
            for joint in range(self.__target_handle.shape[1]):
                self.sim.setJointTargetPosition(int(self.__target_handle[leg][joint]),
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
            noise = np.random.uniform(-0.1, 0.1, size=(15, ))
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
        robot_height = self.sim.getObjectPosition(self.sim.getObject('/head'))[2]
        LF_foot_height = self.sim.getObjectPosition(self.sim.getObject('/foot_FL'))[2]
        RF_foot_height = self.sim.getObjectPosition(self.sim.getObject('/foot_FR'))[2]
        return robot_height > 0.0 and LF_foot_height > 0.0 and RF_foot_height > 0.0

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
        # terminated = False
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
    # env.start()
    for i in range(100):
        action = np.random.uniform(-1, 1, size=15)
        obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.get_states()
        # print(f"Step {i+1}, Action: {action}, States: {next_obs}")
    env.stop()
