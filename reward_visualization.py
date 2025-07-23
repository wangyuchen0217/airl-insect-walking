import os
import sys
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from networks.discrim import AIRLDiscrim
from common.normalized_env import CoppeliaSimEnv


def plot_reward_distribution(disc, expert_batch, policy_batch):
    states_exp, _, _, dones_exp, _, next_states_exp = expert_batch
    states_pi, _, _, dones_pi, log_pis, next_states_pi = policy_batch

    with torch.no_grad():
        reward_exp = disc.calculate_reward(states_exp, dones_exp, log_pis=torch.zeros_like(dones_exp), next_states=next_states_exp)
        reward_pi = disc.calculate_reward(states_pi, dones_pi, log_pis, next_states_pi)

    plt.figure(figsize=(8, 5))
    plt.hist(reward_exp.cpu().numpy(), bins=50, alpha=0.5, label='Expert Reward')
    plt.hist(reward_pi.cpu().numpy(), bins=50, alpha=0.5, label='Policy Reward')
    plt.title('Reward Distribution Comparison')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    SAVE_PATH = "logs/Medauroidea/airl/20250718-1409"

    # Set the device and env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OnTimeStep = True
    env = CoppeliaSimEnv(port=23002, OnTimeStep=OnTimeStep)
    
    # Get state and action shapes from the environment
    state_shape = env.observation_space.shape    # e.g., (27,)
    action_shape = env.action_space.shape          # e.g., (8,)
    
    # Instantiate the discriminator network with the same architecture as used during training
    disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=0.995,
            hidden_units_r=(100, 100),
            hidden_units_v=(100, 100),
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)
    
    # Load the saved disc model parameters from a .pth file
    disc_path = f"{SAVE_PATH}/model/discriminator.pth"
    if os.path.exists(disc_path):
        disc.load_state_dict(torch.load(disc_path, weights_only=True, map_location=device))
        print(f"Loaded actor model from {disc_path}")
    else:
        print(f"Actor model file not found: {disc_path}")
        return
    print("---  Discrinimator Networks ---")
    for name, param in disc.named_parameters():
        print(f"{name}: {param.shape}")
    print(f"---  Statistics ---")
    for name, param in disc.named_parameters():
        if param.requires_grad:
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            l2_norm = param.data.norm(2).item()
            print(f"{name}: mean={mean_val:.4f}, std={std_val:.4f}, L2 norm={l2_norm:.4f}")
        
    # visualize the reward function
    num_points = 100
    state_grid = np.linspace(-1, 1, num_points)
    action_grid = np.linspace(-1, 1, num_points)
    rewards = np.zeros((num_points, num_points))
    for i, state in enumerate(state_grid):
        for j, action in enumerate(action_grid):
            state = np.zeros(state_shape[0])
            state[0] = s
            state[1] = a
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            reward = disc.calculate_reward(
                states=state_tensor.unsqueeze(0),
                dones=torch.zeros(1, 1).to(device),
                log_pis=torch.zeros(1, 1).to(device),
                next_states=state_tensor.unsqueeze(0)
            )
            rewards[i, j] = reward
    rewards = rewards.reshape(num_points, num_points)
    plt.figure(figsize=(8, 6))
    plt.imshow(rewards, extent=(-1, 1, -1, 1), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Reward')
    plt.title('Reward Function Visualization')
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.show()



if __name__ == "__main__":
    main()
