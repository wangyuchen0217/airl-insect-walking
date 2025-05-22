import torch
from torch import nn
import torch.nn.functional as F
from networks.utils import build_mlp

class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))
        

'''Oriniginal AIRLDiscrim with dropout'''
class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(32, 32), # （64, 64）
                 hidden_units_v=(32, 32), # （64, 64）
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        def build_dropout_mlp(input_dim, output_dim, hidden_units, hidden_activation, dropout_p=0.3):
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_units:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(hidden_activation)
                layers.append(nn.Dropout(p=dropout_p))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)

        self.g = build_dropout_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r,
            dropout_p=0.3
        )
        self.h = build_dropout_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v,
            dropout_p=0.3
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)

'''Oriniginal AIRLDiscrim without dropout'''
# class AIRLDiscrim(nn.Module):

#     def __init__(self, state_shape, gamma,
#                  hidden_units_r=(64, 64),
#                  hidden_units_v=(64, 64), 
#                  hidden_activation_r=nn.ReLU(inplace=True),
#                  hidden_activation_v=nn.ReLU(inplace=True)):
#         super().__init__()

#         self.g = build_mlp(
#             input_dim=state_shape[0],
#             output_dim=1,
#             hidden_units=hidden_units_r,
#             hidden_activation=hidden_activation_r
#         )
#         self.h = build_mlp(
#             input_dim=state_shape[0],
#             output_dim=1,
#             hidden_units=hidden_units_v,
#             hidden_activation=hidden_activation_v
#         )

#         self.gamma = gamma

#     def f(self, states, dones, next_states):
#         rs = self.g(states)
#         vs = self.h(states)
#         next_vs = self.h(next_states)
#         return rs + self.gamma * (1 - dones) * next_vs - vs

#     def forward(self, states, dones, log_pis, next_states):
#         # Discriminator's output is sigmoid(f - log_pi).
#         return self.f(states, dones, next_states) - log_pis

#     def calculate_reward(self, states, dones, log_pis, next_states):
#         with torch.no_grad():
#             logits = self.forward(states, dones, log_pis, next_states)
#             return -F.logsigmoid(-logits)


'''Oriniginal AIRLDiscrim with GRU'''
# class AIRLDiscrim(nn.Module):

#     def __init__(self, state_shape, gamma, hidden_size=64):
#         super().__init__()

#         self.gamma = gamma
#         self.state_dim = state_shape[0]

#         # GRU-based sequence encoder for reward and value
#         self.rnn_r = nn.GRU(self.state_dim, hidden_size, batch_first=True)
#         self.rnn_v = nn.GRU(self.state_dim, hidden_size, batch_first=True)

#         self.fc_r = nn.Linear(hidden_size, 1)
#         self.fc_v = nn.Linear(hidden_size, 1)

#     def f(self, states, dones, next_states):
#         # states, next_states: (B, T, D), dones: (B, T, 1)
#         _, h_t = self.rnn_r(states)       # h_t: (1, B, H)
#         _, h_tp1 = self.rnn_v(next_states)

#         r_t = self.fc_r(h_t.squeeze(0))   # (B, 1)
#         v_t = self.fc_v(h_t.squeeze(0))
#         v_tp1 = self.fc_v(h_tp1.squeeze(0))

#         dones = dones.view(-1, 1)   # (B, 1)
#         return r_t + self.gamma * (1 - dones[:, -1]) * v_tp1 - v_t

#     def forward(self, states, dones, log_pis, next_states):
#         # log_pis: (B, 1)
#         return self.f(states, dones, next_states) - log_pis

#     def calculate_reward(self, states, dones, log_pis, next_states):
#         with torch.no_grad():
#             logits = self.forward(states, dones, log_pis, next_states)
#             return -F.logsigmoid(-logits)
