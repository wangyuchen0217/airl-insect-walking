import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
from algorithms.generator import ppo_update
from expert import collect_agent_trajectories


def train_airl(env, policy, num_iterations, steps_per_iter, discriminator_iters, batch_size, discriminator, 
                            expert_loader, expert_dataset, disc_optimizer, policy_optimizer, device):
    
    # Main training loop
    for iteration in range(num_iterations):
        # 1. Collect agent trajectories
        agent_traj = collect_agent_trajectories(env, policy, steps_per_iter, device)
        
        # 2. Update the discriminator several times
        for _ in range(discriminator_iters):
            # Sample a batch from agent data
            agent_indices = np.random.choice(len(agent_traj), batch_size, replace=True)
            agent_states = torch.FloatTensor(np.array([agent_traj[i]['state'] for i in agent_indices])).to(device)
            agent_actions = torch.FloatTensor(np.array([agent_traj[i]['action'] for i in agent_indices])).to(device)
            agent_next_states = torch.FloatTensor(np.array([agent_traj[i]['next_state'] for i in agent_indices])).to(device)
            g_agent = discriminator(agent_states, agent_actions, agent_next_states)
            # For agent data, the target label is 0.
            loss_agent = nn.BCEWithLogitsLoss()(g_agent, torch.zeros_like(g_agent))
            
            # Sample a batch from expert data
            try:
                expert_batch = next(iter(expert_loader))
            except StopIteration:
                expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
                expert_batch = next(iter(expert_loader))
            expert_states_batch, expert_actions_batch, expert_next_states_batch = expert_batch
            g_expert = discriminator(expert_states_batch, expert_actions_batch, expert_next_states_batch)
            # For expert data, the target label is 1.
            loss_expert = nn.BCEWithLogitsLoss()(g_expert, torch.ones_like(g_expert))
            
            # Total discriminator loss and optimization step
            disc_loss = loss_agent + loss_expert
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
        
        # 3. Compute rewards for agent transitions using the discriminator.
        # (Note: using the identity r = log(D) - log(1-D) = g(s,a,s') ).
        agent_rewards = []
        for traj in agent_traj:
            s = torch.FloatTensor(traj['state']).unsqueeze(0).to(device)
            a = torch.FloatTensor(traj['action']).unsqueeze(0).to(device)
            s_next = torch.FloatTensor(traj['next_state']).unsqueeze(0).to(device)
            with torch.no_grad():
                g_val = discriminator(s, a, s_next)
            agent_rewards.append(g_val.item())
        
        # 4. Update the policy using PPO with the computed (AIRL) rewards
        ppo_update(policy, agent_traj, agent_rewards, policy_optimizer, device)
        
        # Logging: print every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Disc Loss: {disc_loss.item():.4f}")
    
    # Save trained models
    torch.save(policy.state_dict(), "trained_models/policy.pth")
    torch.save(discriminator.state_dict(), "trained_models/discriminator.pth")
    print("Saved trained policy and discriminator models.")
    
    env.close()
    return policy, discriminator

def visualize_trained_agent(policy, device, env_name="HalfCheetah-v4", episodes=5, sleep_time=0.02):
    """
    Create a Gymnasium Mujoco environment with rendering enabled (via render_mode='human')
    and run a few episodes using the trained policy.
    """
    # Create the environment with rendering enabled.
    env = gym.make(env_name, render_mode='human')
    policy.eval()  # Set the policy in evaluation mode.
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Render the environment. This should open a window for Mujoco.
            env.render()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.act(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward
            time.sleep(sleep_time)  # slow down the simulation for visualization
        print(f"Episode {ep+1} reward: {total_reward}")
    env.close()