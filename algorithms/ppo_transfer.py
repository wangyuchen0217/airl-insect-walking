import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from common.buffer import RolloutBuffer
from common.base import Algorithm
from networks.actor import ActorNetworkPolicy
from networks.critic import CriticNetworkPolicy
from networks.discrim import AIRLDiscrim
import os


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
    def __init__(self,
                 state_shape,
                 action_shape,
                 device,
                 seed,
                 gamma=0.995,
                 rollout_length=2048,
                 mix_buffer=20,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 units_actor=(64, 64),
                 units_critic=(64, 64),
                 epoch_ppo=10,
                 clip_eps=0.2,
                 lambd=0.97,
                 coef_ent=0.0,
                 max_grad_norm=10.0,
                 mini_batch_size=64
                 ):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.device = device

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor
        self.actor = ActorNetworkPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic
        self.critic = CriticNetworkPolicy(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        # 训练超参 & 状态
        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size


        self.disc = None
        airl_disc_path = 'logs/Medauroidea_66k_aug3c/airl_logit_vx/20251101-1111/model/step460000/discriminator.pth'
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=(100,100),
            hidden_units_v=(100,100),
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True),
        ).to(device)

        state_dict = torch.load(airl_disc_path, map_location=device)
        self.disc.load_state_dict(state_dict)
        self.disc.eval()
        print(f"[INFO] Loaded AIRL discriminator from {airl_disc_path}")

    @torch.no_grad()
    def airl_reward_g(self, state):
        """
        use the learned AIRL discriminator to compute reward
        state: numpy array 或 torch tensor, shape = [obs_dim]
        """
        if isinstance(state, torch.Tensor):
            s = state.to(self.device).unsqueeze(0)  # [1, obs_dim]
        else:
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        r = self.disc.g(s)  # [1, 1]
        return float(r.item())

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1
        action, log_pi = self.explore(state)
        next_state, env_reward, done, truncated, info = env.step(action)
        done = done or truncated
        mask = False if t == env._max_episode_steps else done
        # ==== Use AIRL discriminator to compute reward ====
        reward = self.airl_reward_g(next_state) + env_reward
        self.buffer.append(state, action, reward, mask, log_pi, next_state)
        if done:
            t = 0
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                next_state, info = reset_out
            else:
                next_state = reset_out
            if isinstance(next_state, dict):
                next_state = next_state.get("observation", next_state)
        return next_state, t

    def update(self, writer=None, model_dir=None):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, writer=None):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(values, rewards, dones, next_values, self.gamma, self.lambd)

        # 这里未做 mini-batch 打乱；如需小批量，可按需要切分 states 等张量
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer=None):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)
            writer.add_scalar(
                'gae/target_mean', targets.mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/target_std', targets.std().item(), self.learning_steps)
            writer.add_scalar(
                'gae/value_mean', self.critic(states).mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/value_std', self.critic(states).std().item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)
            writer.add_scalar(
                'stats/ratio', ratios.mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/gae_mean', gaes.mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/gae_std', gaes.std().item(), self.learning_steps)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
