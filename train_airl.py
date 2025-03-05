import gymnasium as gym
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from algorithms.generator import PolicyNetwork
from algorithms.discriminator import AIRLDiscriminator
from expert import load_expert_data
from airl import train_airl, visualize_trained_agent

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'Ant'
ENV_NAME = model_name+"-v4"  # Use a Mujoco-based Gymnasium environment
num_iterations = 1000        # Total AIRL iterations (adjust as needed)
steps_per_iter = 2048        # Number of agent steps per iteration
batch_size = 64              # Mini-batch size for discriminator updates
discriminator_iters = 5      # How many discriminator updates per iteration
gamma = 0.99                 # Discount factor

# Create the environment (for training)
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print("state_dim:", state_dim, "action_dim:", action_dim)

# Initialize policy and its optimizer
policy = PolicyNetwork(state_dim, action_dim).to(device)
policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# Initialize the AIRL discriminator and its optimizer
discriminator = AIRLDiscriminator(state_dim, action_dim, gamma=gamma).to(device)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# Load expert data (replace with your expert trajectories)
expert_states, expert_actions, expert_next_states = load_expert_data(model_name)
expert_dataset = TensorDataset(torch.FloatTensor(expert_states).to(device),
                                torch.FloatTensor(expert_actions).to(device),
                                torch.FloatTensor(expert_next_states).to(device))
expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)

# Train the models and save them.
trained_policy, trained_discriminator = train_airl(env, policy, num_iterations, steps_per_iter, discriminator_iters, batch_size, discriminator, 
                            expert_loader, expert_dataset, disc_optimizer, policy_optimizer, device)

# trained_policy = PolicyNetwork(state_dim, action_dim).to(device)
# trained_policy.load_state_dict(torch.load("trained_models/policy.pth", map_location=device))

# # Visualize the trained policy in Mujoco
# visualize_trained_agent(trained_policy, device, ENV_NAME)

# test aa