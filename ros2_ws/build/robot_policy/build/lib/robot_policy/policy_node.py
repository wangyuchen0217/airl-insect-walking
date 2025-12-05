import time
from typing import List, Optional

import sys
import os
current_file = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(current_file, "../../../../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import rclpy
from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
import torch
import math
from networks.actor import ActorNetworkPolicy

def quat_to_euler(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) to Euler angles roll, pitch, yaw (radians)
    with rotation order XYZ (roll, pitch, yaw)
    """
    # roll (x)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    # pitch (y)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    # yaw (z)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

class PolicyNode(Node):
    """
    Subscribe to /joint_states and construct obs in the same order as during training;
    Use a fixed-weight policy to obtain actions (target joint angles);
    Publish to the position controller topic /position_controller/commands (Float64MultiArray).

    Key parameters (set via ROS parameter server or launch file):
      - joint_names: List of joint names (order determines action vector arrangement)
      - rate_hz: Control frequency
      - model_path: TorchScript model path (optional; if not using torch, a sample MLP is used)
      - obs_mean_path / obs_std_path: Observation normalization statistics (npy, optional)
      - output_mode: 'absolute' | 'delta'  # absolute angle or incremental angle
      - joint_min / joint_max: Soft limits for each joint (same length as joint_names)
      - max_delta: Maximum per-step increment (rad), for smoothing and safety
      - ramp_up_sec: Ramp-up duration (seconds)
    """

    def __init__(self):
        super().__init__("policy_node")

        # ===== actor model loading config ===== 
        ENV_ID = "RedMirror_66k_aug3c"
        ALGO = "ppo-transfer"
        FILENAME = "20251115-2125" 
        CUDA = 0
        STEP_NUM = 790000  # Choose a certain step number of the saved model or None 
        actor_path = os.path.join("/home/yuchen/airl-insect-walking/logs", ENV_ID, ALGO, FILENAME, "model", f"step{STEP_NUM}", "actor.pth")
        # device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
        device = torch.device("cpu")

        # === Parameters ===
        self.joint_names: List[str] = self.declare_parameter(
            "joint_names", ["LF_ThC", "LF_CTr", "LF_FTi",
                                            "LM_ThC", "LM_CTr", "LM_FTi",
                                            "LH_ThC", "LH_CTr", "LH_FTi",
                                            "RF_ThC", "RF_CTr", "RF_FTi",
                                            "RM_ThC", "RM_CTr", "RM_FTi",
                                            "RH_ThC", "RH_CTr", "RH_FTi"]
        ).get_parameter_value().string_array_value

        self.dxl_ids = [
              11,12,13, # LF
              31,32,33, # LM
              51,52,53, # LH
              21,22,23, # RF
              41,42,43, # RM
              61,62,63] # RH

        # ===== Direction mapping between real robot and training env =====
        # q_env = dir_conv * q_robot ; a_robot = dir_conv * a_env
        self.dir_conv = np.array([
            [-1.0,  1.0,  1.0],  # LF_ThC, LF_CTr, LF_FTi
            [-1.0,  1.0,  1.0],  # LM_ThC, LM_CTr, LM_FTi
            [-1.0,  1.0,  1.0],  # LH_ThC, LH_CTr, LH_FTi
            [-1.0,  1.0,  1.0],  # RF_ThC, RF_CTr, RF_FTi
            [-1.0,  1.0,  1.0],  # RM_ThC, RM_CTr, RM_FTi
            [-1.0,  1.0,  1.0],  # RH_ThC, RH_CTr, RH_FTi
        ], dtype=np.float32).reshape(-1)  # (18,)

        # ===== Initial joint offsets (consistent with CoppeliaSim environment) =====
        self.init_pos_deg = np.array([
            [ 30.0,   9.5, -60.0],
            [  0.0,  -2.5, -60.0],
            [-40.0,   9.5, -60.0],
            [ 30.0,   9.5, -60.0],
            [  0.0,  -2.5, -60.0],
            [-40.0,   9.5, -60.0],
        ], dtype=np.float32)

        self.init_pos_dir = np.array([
            [1.0,  1.0,  1.0],
            [1.0,  1.0,  1.0],
            [1.0,  1.0,  1.0],
            [ -1.0, -1.0, -1.0],
            [ -1.0, -1.0, -1.0],
            [ -1.0, -1.0, -1.0],
        ], dtype=np.float32)

        self.imu_offset = None

        # ===== Initial joint positions in radians =====
        init_pos_deg_signed = self.init_pos_deg * self.init_pos_dir
        self.init_joint_position = np.deg2rad(init_pos_deg_signed).astype(np.float32).reshape(-1)  # (18,)
        self.last_cmd = self.init_joint_position.copy()

        # ===== Sim to real calibration =====
        self.mu_sim = np.load("/home/yuchen/airl-insect-walking/ros2_ws/rosbag/rosbag2_2025_11_24-21_52_50/mu_sim.npy").astype(np.float32)
        self.sigma_sim = np.load("/home/yuchen/airl-insect-walking/ros2_ws/rosbag/rosbag2_2025_11_24-21_52_50/sigma_sim.npy").astype(np.float32)
        self.mu_real = np.load("/home/yuchen/airl-insect-walking/ros2_ws/rosbag/rosbag2_2025_11_24-21_52_50/mu_real.npy").astype(np.float32)
        self.sigma_real = np.load("/home/yuchen/airl-insect-walking/ros2_ws/rosbag/rosbag2_2025_11_24-21_52_50/sigma_real.npy").astype(np.float32)
        
        assert self.mu_sim.shape[0] == 21
        assert self.mu_real.shape[0] == 21
        
        # to avoid division by zero
        self.sigma_real_safe = np.where(self.sigma_real < 1e-6, 1.0, self.sigma_real)


        self.rate_hz = self.declare_parameter("rate_hz", 30).value
        # ===== Observation normalization stats ===== 
        self.model_path = self.declare_parameter("model_path", actor_path).value
        self.imu_topic = self.declare_parameter("imu_topic", "/red_mirror/imu").value
        self.joint_states = self.declare_parameter("joint_states_topic", "/red_mirror/DXL_cur_positions").value
        self.contact_topic = self.declare_parameter("contact_topic", "/red_mirror/foot_contact").value
        # =====================================
        self.output_mode = self.declare_parameter("output_mode", "offset").value # or 'delta'
        self.max_delta = self.declare_parameter("max_delta", 0.02).value # rad/step
        self.ramp_up_sec = self.declare_parameter("ramp_up_sec", 2.0).value


        # Soft joint limits (example: each joint -90° ~ +90°)
        default_min = [-1.57] * len(self.joint_names)
        default_max = [ +1.57] * len(self.joint_names)
        self.joint_min = np.array(self.declare_parameter("joint_min", default_min).value, dtype=np.float32)
        self.joint_max = np.array(self.declare_parameter("joint_max", default_max).value, dtype=np.float32)

        # Subscribers & Publishers
        # self.sub_js = self.create_subscription(JointState, "/joint_states", self.on_joint_state, 50)
        self.sub_js = self.create_subscription(Float64MultiArray, "/red_mirror/DXL_cur_positions", self.on_joint_state, 50)
        self.sub_imu = self.create_subscription(Float64MultiArray, self.imu_topic, self.on_imu, 10)
        self.sub_contact = self.create_subscription(Float64MultiArray, self.contact_topic, self.on_contact, 10)

        self.pub_cmd = self.create_publisher(Float64MultiArray, "red_mirror/DXL_cmd_ID_positions", 10)

        # State cache
        self.latest_js: Optional[np.ndarray] = None
        self.latest_rpy: Optional[np.ndarray] = None   # (3,) roll, pitch, yaw
        self.latest_contact: Optional[np.ndarray] = None  # (6,)
        self.name_to_index: dict = {}
        # self.last_cmd = np.zeros(len(self.joint_names), dtype=np.float32)
        self.start_time = time.time()

        self.orientation_low = min([-0.1253066, -0.21079601, -0.14037536])
        self.orientation_high = max([0.17421827, 0.03616637, 0.56608814])

        self.joint_low = np.array([
            -1.3860602,  0.06034265, -2.4969175,
            -0.9650939, -0.0351965 , -2.3240883,
             0.28500196, 0.15376441, -2.507828,
             0.5072578, -0.7540891 ,  0.5758628,
            -0.6832747, -0.6967423 ,  0.64540935,
            -1.2671623, -1.0010364 ,  0.58814275
        ], dtype=np.float32)

        self.joint_high = np.array([
            -0.5072578 ,  0.7540891 , -0.5758628,
             0.6832747 ,  0.6967423 , -0.64540935,
             1.2671623 ,  1.0010364 , -0.58814275,
             1.3860602 , -0.06034265,  2.4969175,
             0.9650939 ,  0.0351965 ,  2.3240883,
            -0.28500196, -0.15376441,  2.507828
        ], dtype=np.float32)

        self.action_space_high = np.array([
            -0.08928384, 0.64018328, 0.73880163,
            0.71728384, 0.53050838, 0.52528891,
            0.61509333, 0.76640703, 0.46057537,
            0.87071587, 0.19994925, 1.43173578,
            0.90740824, 0.03776942, 1.30299309,
            0.44833541, 0.00427082, 1.59938887
        ], dtype=np.float32)

        self.action_space_low = np.array([
            -0.87071587, -0.19994925, -1.43173578,
            -0.90740824, -0.03776942, -1.30299309,
            -0.44833541, -0.00427082, -1.59938887,
            0.08928384, -0.64018328, -0.73880163,
            -0.71728384, -0.53050838, -0.52528891,
            -0.61509333, -0.76640703, -0.46057537
        ], dtype=np.float32)

        self.action_mid = (self.action_space_high + self.action_space_low) / 2.0
        self.action_scale = (self.action_space_high - self.action_space_low) / 2.0

        self.policy = None
        if self.model_path and torch is not None:
            try:
                self.policy = ActorNetworkPolicy(
                        state_shape=(27,),          
                        action_shape=(len(self.joint_names),),          
                        hidden_units=(64, 64),
                        hidden_activation=torch.nn.Tanh()
                    ).to(device)
                state_dict = torch.load(self.model_path, weights_only=True, map_location=device)
                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                self.get_logger().info(f"Loaded TorchScript model from: {self.model_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load TorchScript: {e}. Fallback to dummy MLP.")
        if self.policy is None:
            self.get_logger().warn("Using built-in dummy MLP policy (random weights, for connectivity testing only)")
            # Minimal network using numpy (obs->tanh->linear mapping to mid position), for testing only
            self.W = np.random.randn((len(self.joint_names)*2), len(self.joint_names)).astype(np.float32) * 0.01
            self.b = np.zeros(len(self.joint_names), dtype=np.float32)

        # Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

        # Joint index mapping (ensure /joint_states order maps correctly to joint_names order)
        self.name_to_index: dict = {}


    def calibrate_sim2real(self, raw_state: np.ndarray) -> np.ndarray:
        x = raw_state.astype(np.float32).copy()

        # roll_bias = self.mu_real[0] - self.mu_sim[0]  # 例如 0.6 rad
        # x[0] = x[0] - roll_bias              

        calib = (x[:21] - self.mu_real) / self.sigma_real_safe * self.sigma_sim + self.mu_sim

        # ===== mask for selective calibration =====    
        mask = np.zeros(21, dtype=bool)

        # case: calibrate only rpy
        mask[0:3] = True

        # case: calibrate only roll, pitch
        # mask[0:2] = True

        # case: calibrate only  legs' joints
        # mask[3:21] = True

        # case : calibrate roll, pitch and legs' joints
        # mask[0:2] = True
        # mask[3:21] = True


        # case : calibrate only forelegs' joints
        # LF: 3-5, RF: 12-14
        # mask[3:6] = True
        # mask[12:15] = True

        # ==================================

        # case 0: calibrate all
        # mask[:] = True

        # apply calibration mask (case selection)
        x[:21][mask] = calib[mask]

        # # calibration rate
        # alpha_rpy = 0.8
        # alpha_joints = 0.4
        # x[0:3] = (1-alpha_rpy) * raw_state[0:3] + alpha_rpy * calib[0:3]
        # x[3:21] = (1-alpha_joints) * raw_state[3:21] + alpha_joints * calib[3:21]

        return x



    def on_joint_state(self, msg: Float64MultiArray):
        arr = np.array(msg.data, dtype=np.float32).reshape(-1)
        if arr.size < len(self.joint_names):
            self.get_logger().warn(f"DXL_cur_positions len {arr.size} < {len(self.joint_names)}")
            return
        self.latest_js = arr[:len(self.joint_names)]

    def on_imu(self, msg: Float64MultiArray):
        arr = np.array(msg.data, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            self.get_logger().warn(f"IMU msg size {arr.size} < 3")
            return
        # self.latest_rpy = arr[:3]
        rpy_deg = arr[:3]
        rpy_rad = np.deg2rad(rpy_deg).astype(np.float32)
        # Apply IMU offset calibration
        if self.imu_offset is None:
            self.imu_offset = rpy_rad.copy()
            self.get_logger().info(
                f"Set IMU offset (deg): {rpy_deg}, (rad): {self.imu_offset}"
            )
        rpy_centered = rpy_rad - self.imu_offset
        self.latest_rpy = rpy_centered

    def on_contact(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=np.float32).reshape(-1)
        if data.size < 6:
            self.get_logger().warn(f"Contact message size {data.size} < 6")
            return
        # contact: 1.0 if distance < 10.0 else 0.0
        contact = (data[:6] < 10.0).astype(np.float32)
        # raw contact order: [LF,RF,LM,RM,LH,RH] -> desired order: [LF,LM,LH,RF,RM,RH]
        reorder = [0, 2, 4, 1, 3, 5]
        reordered_contact = contact[reorder]
        self.latest_contact = reordered_contact

    def build_obs(self) -> Optional[np.ndarray]:
        if self.latest_js is None or self.latest_rpy is None or self.latest_contact is None:
            return None
        # if not self.name_to_index:
        #     return None
        
        # joint_pos_robot = np.array(
        #     [self.latest_js.position[self.name_to_index[n]] for n in self.joint_names],
        #     dtype=np.float32)

        joint_pos_robot = self.latest_js.astype(np.float32)
        
        # ---- Convert joint positions to env frame ---- #
        joint_pos_env = joint_pos_robot * self.dir_conv  # (18,)

        rpy = self.latest_rpy.astype(np.float32)  # (3,)
        contact = self.latest_contact.astype(np.float32)  # (6,)
        raw_state = np.concatenate([rpy, joint_pos_env, contact], axis=0)

        # ---- Sim2Real calibration ---- #
        raw_state_calib = self.calibrate_sim2real(raw_state)
        # raw_state_calib = raw_state

        # ---- Do as env.normalize_observation---- #
        # orientation: 'shared'
        ori = raw_state_calib[0:3]
        low_o = self.orientation_low
        high_o = self.orientation_high
        norm_ori = 2.0 * (ori - low_o) / (high_o - low_o) - 1.0
        # joint_angles: 'per_dim'
        j = raw_state_calib[3:3+18]
        norm_j = 2.0 * (j - self.joint_low) / (self.joint_high - self.joint_low) - 1.0
        # contact: 'binary' → 0→-1, 1→1
        c = raw_state_calib[3+18:3+18+6]
        norm_c = c * 2.0 - 1.0
        # combine the normalized states
        obs = np.concatenate([norm_ori, norm_j, norm_c], axis=0)  # (27,)

        return obs.astype(np.float32)
    
    def denormalize_action(self, norm_action):
        return norm_action * self.action_scale + self.action_mid

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is not None and torch is not None:
            with torch.no_grad():
                x = torch.from_numpy(obs).unsqueeze(0)  # [1, obs_dim]
                a = self.policy(x).cpu().numpy()[0]
        else:
            # Dummy: a = tanh(obs @ W + b), mapped to joint midpoint
            hid = np.tanh(obs @ self.W + self.b)
            mid = (self.joint_min + self.joint_max) * 0.5
            a = mid + 0.2 * hid  # For safety, output midpoint (no motion); modify as needed
        return a.astype(np.float32)

    def limit_and_ramp(self, target: np.ndarray) -> np.ndarray:
        # Joint limits
        target = np.clip(target, self.joint_min, self.joint_max)
        # Rate limit (max per-cycle delta)
        delta = np.clip(target - self.last_cmd, -self.max_delta, self.max_delta)
        cmd = self.last_cmd + delta
        # Startup ramp
        scale = np.clip((time.time() - self.start_time) / self.ramp_up_sec, 0.0, 1.0)
        return self.last_cmd + (cmd - self.last_cmd) * scale

    def on_timer(self):
        obs = self.build_obs()
        if obs is None:
            return

        # denormalize action
        norm_action = self.run_policy(obs)
        a_env = self.denormalize_action(norm_action) 

        # ---- Convert action to robot frame ---- #
        a_robot = a_env * self.dir_conv  # (18,)

        if self.output_mode == "delta":
            target = self.last_cmd + a_robot
        elif self.output_mode == "offset":
            target = self.init_joint_position + a_robot
        else:
            target = a_robot

        # cmd = self.limit_and_ramp(target)
        cmd = target
        self.last_cmd = cmd.copy()

        # ---- packaging [ID0,pos0, ID1,pos1, ...] ----
        pairs = []
        for dxl_id, angle in zip(self.dxl_ids, cmd):
            pairs.append(float(dxl_id))   # ID
            pairs.append(float(angle))    # cmd

        msg = Float64MultiArray()
        msg.data = pairs
        self.pub_cmd.publish(msg)


def main():
    rclpy.init()
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

