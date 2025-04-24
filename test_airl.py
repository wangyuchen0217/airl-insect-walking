import os
import sys
import gymnasium as gym
import torch
import numpy as np
import envs
from networks.actor import ActorNetworkPolicy 
import logging
from common.base import LoggerWriter
from common.env import make_env

def main():
    SAVE_PATH = "/home/yuchen/airl_insect_walking/logs/StickInsect-v4/airl/20250424-1604"
    ENV_ID = "StickInsect-v4"
    NUM_EPISODES = 10
    SEED = 123

    # Log the evaluation process
    log_filename = os.path.join(SAVE_PATH, "evaluation.log")
    logging.basicConfig(
        filename=log_filename,    
        level=logging.INFO,
        format='%(message)s',
        filemode='w'
        )
    sys.stdout = LoggerWriter(logging.info)

    # Set the device and env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_ID, test=True)
    
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
    actor_path = f"{SAVE_PATH}/model/actor.pth"
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
        while not done:
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
            # print(f"Step: {step}, Reward: {reward:.2f}, Done: {done}")
            
            # Optionally, render is already enabled via render_mode="human"
            # env.render()  # 如果需要手动调用 render，可以取消注释
            
        print(f"Episode {ep+1}: Return = {ep_return:.2f}, Steps = {step}")
    
    env.close()

if __name__ == "__main__":
    main()

