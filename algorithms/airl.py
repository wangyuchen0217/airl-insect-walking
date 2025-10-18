import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from algorithms.ppo import PPO
from networks.discrim import AIRLDiscrim
import sys


class AIRL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(64, 64), units_disc_v=(64, 64),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        # disc_path = 'logs/Medauroidea_60000_fcontact/airl_logit/20250930-1445/model/step860000/discriminator.pth'
        # self.disc.load_state_dict(torch.load(disc_path, weights_only=True, map_location=device))

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc


    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # -------Samples from current policy's trajectories------ #
            states, _, _, dones, log_pis, next_states = \
                self.buffer.sample(self.batch_size)
            # ---------------------------------------------------------------------------- #
            # -------Samples from last k iterations-------- #
            # states, actions, _, dones, log_pis, next_states = \
            #     self.buffer.sample_last_k_iters(self.batch_size, k=20, include_current=True)
            # ------------------------------------------------------------ #
            
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                # -------Recalculate log_pis because of the k-iter sample------ #
                # log_pis = self.actor.evaluate_log_pi(states, actions)
                # -------------------------------------------------------------------------------------- #
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp, actions_exp)
            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )

        # We don't use reward signals here,
        states, actions, env_r, dones, log_pis, next_states = self.buffer.get()

        # ---------- Calculate rewards: -log sigmoid(-logit) ---------- #
        # rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)
        # rewards_exp = self.disc.calculate_reward(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # ---------- Calculate rewards: logit ---------- #
        logits = self.disc(states, dones, log_pis, next_states).detach()
        # vx_100 = env_r
        # rewards = logits + vx_100
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp).detach()
        # dx_exp = next_states_exp[:,0] - states_exp[:,0]
        # rewards_exp = rewards_exp + dx_exp * 100

        # f(s,s')
        f_value = self.disc.f(states, dones, next_states).detach()
        f_value_exp = self.disc.f(states_exp, dones_exp, next_states_exp).detach()
        
        # add debug for logits
        writer.add_scalar(
            'return/logit_mean', logits.mean().item(), self.learning_steps)
        writer.add_scalar(
            'return/logit_std', logits.std().item(), self.learning_steps)
        writer.add_scalar(
            'return/logit_exp_mean', logits_exp.mean().item(), self.learning_steps)
        writer.add_scalar(
            'return/logit_exp_std', logits_exp.std().item(), self.learning_steps)
        # add debug for f values
        writer.add_scalar(
            'return/f_value_mean', f_value.mean().item(), self.learning_steps)
        writer.add_scalar(
            'return/f_value_std', f_value.std().item(), self.learning_steps)
        writer.add_scalar(
            'return/f_value_exp_mean', f_value_exp.mean().item(), self.learning_steps)
        writer.add_scalar(
            'return/f_value_exp_std', f_value_exp.std().item(), self.learning_steps)
        
        # # add debug for reward: losspi
        # writer.add_scalar(
        #     'return/reward_mean', rewards.mean().item(), self.learning_steps)
        # writer.add_scalar(
        #     'return/reward_std', rewards.std().item(), self.learning_steps)
        
        # add debug for g(s) outputs
        g_value = self.disc.g(states).detach()
        g_value_exp = self.disc.g(states_exp).detach()

        writer.add_scalar(
            'disc/g_value_mean', g_value.mean().item(), self.learning_steps)
        writer.add_scalar(
            'disc/g_value_std', g_value.std().item(), self.learning_steps)
        writer.add_scalar(
            'disc/g_value_exp_mean', g_value_exp.mean().item(), self.learning_steps)
        writer.add_scalar(
            'disc/g_value_exp_std', g_value_exp.std().item(), self.learning_steps)

        # add debug for h(s) outputs
        h_value = self.disc.h(states).detach()
        h_value_exp = self.disc.h(states_exp).detach()

        writer.add_scalar(
            'disc/h_value_mean', h_value.mean().item(), self.learning_steps)
        writer.add_scalar(
            'disc/h_value_std', h_value.std().item(), self.learning_steps)
        writer.add_scalar(
            'disc/h_value_exp_mean', h_value_exp.mean().item(), self.learning_steps)
        writer.add_scalar(
            'disc/h_value_exp_std', h_value_exp.std().item(), self.learning_steps)


        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, logits, dones, log_pis, next_states, writer)
        

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # print(f"Update discriminator at step {self.learning_steps_disc}, file=sys.__stdout__")

        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)