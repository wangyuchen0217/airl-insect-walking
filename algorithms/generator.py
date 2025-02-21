import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # Learnable log_std parameter (for Gaussian policies)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        value = self.critic(state)
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value
    
    def act(self, state):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

def ppo_update(policy, trajectories, rewards, optimizer, device, clip_epsilon=0.2,
               ppo_epochs=10, mini_batch_size=64, gamma=0.99, lam=0.95):
    """
    Performs a PPO update using trajectories re-labeled with the AIRL reward.
    """
    # Prepare tensors from trajectories
    states = torch.FloatTensor(np.array([t['state'] for t in trajectories])).to(device)
    actions = torch.FloatTensor(np.array([t['action'] for t in trajectories])).to(device)
    old_log_probs = torch.FloatTensor(np.array([t['log_prob'] for t in trajectories])).to(device)
    
    # Get value estimates from the critic
    with torch.no_grad():
        _, _, values = policy(states)
        values = values.squeeze().cpu().numpy()
    
    # Compute returns and advantages via Generalized Advantage Estimation (GAE)
    rewards = np.array(rewards)
    dones = np.array([t['done'] for t in trajectories], dtype=np.float32)
    returns = []
    advantages = []
    gae = 0
    next_value = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])
    advantages = torch.FloatTensor(advantages).to(device)
    returns = torch.FloatTensor(returns).to(device)
    
    # Create mini-batches
    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    
    for _ in range(ppo_epochs):
        for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
            mean, std, values_pred = policy(batch_states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values_pred.squeeze(), batch_returns)
            loss = actor_loss + 0.5 * critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()