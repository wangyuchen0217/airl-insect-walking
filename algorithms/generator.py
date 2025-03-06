import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from algorithms.utils import build_mlp, reparameterize, evaluate_log_pi

class PolicyNetwork(nn.Module):
    """
    A combined actor-critic network.
    Actor: maps states to action means (Gaussian policy).
    Critic: maps states to state-value estimates.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        # Actor network.
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        # Critic network.
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # Log standard deviation for Gaussian policy (learnable).
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """
        Returns action distribution parameters and value estimate.
        """
        value = self.critic(state)
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value
    
    def act(self, state):
        """
        Samples an action for the given state.
        Returns: action, log_prob, value.
        """
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
    

class ActorNetworkPolicy(nn.Module):
    """
    The actor network: maps states to actions.
    Uses a generic MLP to compute a policy mean and maintains a learnable log standard
    deviation (log_stds) for a Gaussian policy.
    """
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super(ActorNetworkPolicy, self).__init__()
        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        # log_stds is a learnable parameter that defines the standard deviation for all actions.
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        """
        Compute the policy mean given states.
        A tanh activation is applied to bound the outputs.
        """
        mean = torch.tanh(self.net(states))
        return mean

    def sample(self, states):
        """
        Sample actions from the policy distribution using the reparameterization trick.
        Returns a tuple: (action, log probability).
        """
        action, log_prob = reparameterize(self.net(states), self.log_stds)
        return action, log_prob

    def evaluate_log_pi(self, states, actions):
        """
        Evaluate the log probabilities for given actions under the current policy.
        """
        return evaluate_log_pi(self.net(states), self.log_stds, actions)


class CriticNetworkPolicy(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return self.net(states)
    

def ppo_update(policy, trajectories, rewards, optimizer, device,
               clip_epsilon=0.2, ppo_epochs=15, mini_batch_size=64,
               gamma=0.99, lam=0.95, coef_entropy=0.01, max_grad_norm=0.5):
    """
    Performs a PPO update with trajectories that have been re-labeled with the AIRL reward.
    
    Args:
        policy: the PolicyNetwork to update.
        trajectories: a list of dictionaries containing 'state', 'action', 'log_prob', and 'done'.
        rewards: a list/array of rewards computed from the AIRL discriminator.
        optimizer: the optimizer for the policy network.
        device: the device (cpu or cuda).
        clip_epsilon: PPO clipping parameter.
        ppo_epochs: number of epochs to iterate over the rollout.
        mini_batch_size: mini-batch size.
        gamma: discount factor.
        lam: GAE lambda.
        coef_entropy: coefficient for the entropy bonus.
        max_grad_norm: maximum norm for gradient clipping.
    """
    # Convert trajectory data to tensors.
    states = torch.tensor(np.array([t['state'] for t in trajectories]), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array([t['action'] for t in trajectories]), dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(np.array([t['log_prob'] for t in trajectories]), dtype=torch.float32, device=device)
    dones = np.array([t['done'] for t in trajectories], dtype=np.float32)

    # Get value estimates from the critic (detach to numpy for advantage computation).
    with torch.no_grad():
        _, _, values = policy(states)
        values = values.squeeze().cpu().numpy()

    # Compute returns and advantages using GAE.
    returns = []
    advantages = []
    gae = 0
    next_value = 0  # Assumes terminal state has value 0.
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])
    
    # Convert returns and advantages to tensors.
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    # Normalize advantages (standard practice for variance reduction).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Create a DataLoader for mini-batch updates.
    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    
    # PPO update loop.
    for _ in range(ppo_epochs):
        for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
            mean, std, values_pred = policy(batch_states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            # Compute the probability ratio (new / old).
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            # Surrogate losses.
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            # Critic loss (value function loss).
            critic_loss = F.mse_loss(values_pred.squeeze(), batch_returns)
            # Entropy bonus.
            entropy = dist.entropy().sum(dim=-1).mean()
            # Total loss.
            loss = actor_loss + 0.5 * critic_loss - coef_entropy * entropy
            
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to avoid explosion.
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()