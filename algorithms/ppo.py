import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from common.buffer import RolloutBuffer
from common.base import Algorithm
from networks.actor import ActorNetworkPolicy
from networks.critic import CriticNetworkPolicy
    

def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)
    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0, mini_batch_size=64):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        
        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )
        
        # Actor.
        self.actor = ActorNetworkPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)
        
        # Critic.
        self.critic = CriticNetworkPolicy(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)
        
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        
        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, truncated,  info = env.step(action)
        done = done or truncated
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, log_pi, next_state)
        if done:
            t = 0
            # Properly reset the environment.
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                next_state, info = reset_out
            else:
                next_state = reset_out
            # If observation is a dict, extract the "observation" key.
            if isinstance(next_state, dict):
                next_state = next_state.get("observation", next_state)
        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, writer):
        # Compute value estimates.
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
        # Squeeze rewards and dones to have shape (rollout_length,)
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        # Calculate targets and advantages using GAE.
        targets, gaes = calculate_gae(values, rewards, dones, next_values, self.gamma, self.lambd)
        # Create DataLoader for mini-batch updates.
        dataset = TensorDataset(states, actions, log_pis, targets, gaes)
        loader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)
        # PPO update loop.
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            for batch_states, batch_actions, batch_old_log_probs, batch_targets, batch_gaes in loader:
                self.update_critic(batch_states, batch_targets, writer)
                self.update_actor(batch_states, batch_actions, batch_old_log_probs, batch_gaes, writer)

    def update_critic(self, states, targets, writer):
        values_pred = self.critic(states).squeeze()
        loss_critic = F.mse_loss(values_pred, targets)
        self.optim_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()
        ratios = (log_pis - log_pis_old).exp()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        total_actor_loss = loss_actor - self.coef_ent * entropy
        self.optim_actor.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar('stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        # Add saving functionality if needed.
        pass