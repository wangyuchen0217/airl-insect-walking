import os
import sys
import torch
import numpy as np
from networks.actor import ActorNetworkPolicy 
import logging
from common.base import LoggerWriter

coppelia_python_client_path = os.path.expanduser(
    '~/CoppeliaSim/programming/zmqRemoteApi/clients/python'
)
if coppelia_python_client_path not in sys.path:
    sys.path.append(coppelia_python_client_path)
# from common.normalized_env_66k import CoppeliaSimEnv
from common.normalized_env_red_mirror import CoppeliaSimEnv
# from common.normalized_env_66k_legloss import CoppeliaSimEnv
# from common.normalized_env_66k_RM_error import CoppeliaSimEnv
# import tensorboard

import rclpy
from std_msgs.msg import Float64MultiArray

# ======== Parameters (modify these as needed) =========
ENV_ID = "RedMirror_66k_aug3c"
ALGO = "ppo-transfer"
FILENAME = "20251115-2125" 
PORT = 23000 # CoppeliaSim port: default is 23000
CUDA = 0
NUM_EPISODES = 1
STEP_NUM = 790000  # Choose a certain step number of the saved model or None 
LOG = False
# =================================================

# ===== Direction mapping between real robot and training env =====
# q_env = dir_conv * q_robot ; a_robot = dir_conv * a_env
dir_conv = np.array([
            [-1.0,  1.0,  1.0],  # LF_ThC, LF_CTr, LF_FTi
            [-1.0,  1.0,  1.0],  # LM_ThC, LM_CTr, LM_FTi
            [-1.0,  1.0,  1.0],  # LH_ThC, LH_CTr, LH_FTi
            [-1.0,  1.0,  1.0],  # RF_ThC, RF_CTr, RF_FTi
            [-1.0,  1.0,  1.0],  # RM_ThC, RM_CTr, RM_FTi
            [-1.0,  1.0,  1.0],  # RH_ThC, RH_CTr, RH_FTi
        ], dtype=np.float32).reshape(-1)  # (18,)

# ===== Initial joint offsets (consistent with CoppeliaSim environment) =====
init_pos_deg = np.array([
            [ 30.0,   9.5, -60.0],
            [  0.0,  -2.5, -60.0],
            [-40.0,   9.5, -60.0],
            [ 30.0,   9.5, -60.0],
            [  0.0,  -2.5, -60.0],
            [-40.0,   9.5, -60.0],
        ], dtype=np.float32)

init_pos_dir = np.array([
            [1.0,  1.0,  1.0],
            [1.0,  1.0,  1.0],
            [1.0,  1.0,  1.0],
            [ -1.0, -1.0, -1.0],
            [ -1.0, -1.0, -1.0],
            [ -1.0, -1.0, -1.0],
        ], dtype=np.float32)

# ===== Initial joint positions in radians =====
init_pos_deg_signed = init_pos_deg * init_pos_dir
init_joint_position = np.deg2rad(init_pos_deg_signed).astype(np.float32).reshape(-1)  # (18,)
last_cmd = init_joint_position.copy()

def main():
    SAVE_PATH = os.path.join("logs", ENV_ID, ALGO, FILENAME)

    # Log the evaluation process
    if LOG:
        log_filename = os.path.join(SAVE_PATH, "evaluation.log")
        logging.basicConfig(
            filename=log_filename,    
            level=logging.INFO,
            format='%(message)s',
            filemode='w'
            )
        sys.stdout = LoggerWriter(logging.info)

    # ======== ROS2 初始化（新增） ========
    rclpy.init(args=None)
    node = rclpy.create_node('coppeliasim_runner')

    # 发布仿真 state 的话题，比如 /sim_states
    state_pub = node.create_publisher(Float64MultiArray, '/sim_states', 10)

    # 发布 action 的话题，比如 /sim_actions（将来机器人可以订阅这个）
    action_pub = node.create_publisher(Float64MultiArray, '/sim_actions', 10)

    dxl_pub = node.create_publisher(Float64MultiArray, "red_mirror/DXL_cmd_ID_positions", 10)
    # ==================================

    # Set the device and env
    device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
    OnTimeStep = True
    env = CoppeliaSimEnv(port=PORT, OnTimeStep=OnTimeStep)
    
    # Get state and action shapes from the environment
    state_shape = env.observation_space.shape    # e.g., (27,)
    action_shape = env.action_space.shape          # e.g., (8,)
    
    # Instantiate the Actor network with the same architecture as used during training
    actor = ActorNetworkPolicy(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_units=(64, 64),
        hidden_activation=torch.nn.Tanh()
    ).to(device)
    
    # Load the saved actor model parameters from a .pth file
    if STEP_NUM is None:
        actor_path = f"{SAVE_PATH}/model/actor.pth"
    else:
        # actor_path = f"{SAVE_PATH}/model/step{STEP_NUM}/actor.pth"
        actor_path = os.path.join("/home/yuchen/airl-insect-walking/logs", ENV_ID, ALGO, FILENAME, "model", f"step{STEP_NUM}", "actor.pth")
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, weights_only=True, map_location=device))
        print(f"Loaded actor model from {actor_path}")
    else:
        print(f"Actor model file not found: {actor_path}")
        return
    print("---  Actor Networks ---")
    for name, param in actor.named_parameters():
        print(f"{name}: {param.shape}")
    print(f"---  Statistics ---")
    for name, param in actor.named_parameters():
        if param.requires_grad:
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            l2_norm = param.data.norm(2).item()
            print(f"{name}: mean={mean_val:.4f}, std={std_val:.4f}, L2 norm={l2_norm:.4f}")
    
    # Set the model to evaluation mode
    actor.eval()
    
    print(f"---  Evaluation ---")
    for ep in range(NUM_EPISODES):
        # Reset the environment with a fixed seed for reproducibility: seed = SEED
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out
        
        # If observation is a dict, extract the "observation" key
        if isinstance(state, dict):
            state = state.get("observation", state)
        
        done = False
        ep_return = 0.0
        step = 0
        states = []
        actions = []
        velocities = []
        foot_trajs = []
        foot_names = ['/foot_FL', '/foot_ML', '/foot_HL', '/foot_FR', '/foot_MR', '/foot_HR']
        RH_joints = []
        RH_names = ['/m1_HR', '/m2_HR', '/m3_HR']
        while not done:
            # ---------- 发布当前 state 到 ROS2（新增） ----------
            state_array = np.array(state, dtype=np.float32)
            state_msg = Float64MultiArray()
            state_msg.data = state_array.tolist()
            state_pub.publish(state_msg)
            # ------------------------------------------------

            # Convert state to a torch tensor and add batch dimension
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get the action from the actor (deterministic, using the mean)
            with torch.no_grad():
                mean = actor(state_tensor)  # Actor returns the policy mean after tanh activation
                action = mean.cpu().numpy()[0]
            
            # Take a step in the environment
            next_step = env.step(action)
            state, reward, terminated, truncated, info = next_step
            done = terminated or truncated
            ep_return += reward
            step += 1 

            # ---------- 发布当前 action 到 ROS2（新增） ----------
            action_cmd = env.denormalize_action(action) 
            action_cmd = action_cmd * dir_conv  + last_cmd # 转换方向并加上上次命令作为偏移

            action_msg = Float64MultiArray()
            action_msg.data = action_cmd.tolist()
            action_pub.publish(action_msg)
            # 将来你可以让机器人订阅 /sim_actions 来执行动作

            dxl_ids = [11,12,13, # LF
              31,32,33, # LM
              51,52,53, # LH
              21,22,23, # RF
              41,42,43, # RM
              61,62,63]  # RH

            # 将 action 转换成 [ID0,pos0,ID1,pos1,...]
            pairs = []
            for dxl_id, angle in zip(dxl_ids, action_cmd):
                pairs.append(float(dxl_id))
                pairs.append(float(angle))

            dxl_msg = Float64MultiArray()
            dxl_msg.data = pairs
            dxl_pub.publish(dxl_msg)
            # ---------------------------------------------------

            foot_traj = []
            for i in range(6):
                foot_traj_sig = env.sim.getObjectPosition(env.sim.getObject(foot_names[i]))[2]
                foot_traj.append(foot_traj_sig)

            # RH_joint = []
            # for i in range(3):
            #     RH_joint_sig = env.sim.getJointPosition(env.sim.getObject(RH_names[i]))
            #     RH_joint.append(RH_joint_sig)
            
            states.append(state)
            actions.append(action)
            velocities.append(reward)
            foot_trajs.append(foot_traj)
            # RH_joints.append(RH_joint)
        
        if LOG:
            states_array = np.array(states)
            actions_array = np.array(actions)
            os.makedirs(os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}"), exist_ok=True)
            # save as csv
            np.savetxt(os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}", f"episode_{ep+1}_states.csv"), states_array, delimiter=',')
            np.savetxt(os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}", f"episode_{ep+1}_actions.csv"), actions_array, delimiter=',')
            np.savetxt(os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}", f"episode_{ep+1}_velocities.csv"), np.array(velocities), delimiter=',')
            np.savetxt(os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}", f"episode_{ep+1}_foot_trajs.csv"), np.array(foot_trajs), delimiter=',')
            # np.savetxt(os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}", f"episode_{ep+1}_RH_joints.csv"), np.array(RH_joints), delimiter=',')
        print(f"Episode {ep+1}: Return = {ep_return:.2f}, Steps = {step}")
    
    env.stop()

    # ======== ROS2 结束（新增） ========
    node.destroy_node()
    rclpy.shutdown()
    # ===================================

if __name__ == "__main__":
    main()

