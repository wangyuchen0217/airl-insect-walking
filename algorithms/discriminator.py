import torch
import torch.nn as nn

class AIRLDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, gamma=0.99):
        super(AIRLDiscriminator, self).__init__()
        self.gamma = gamma
        # f(s,a) network: learns the immediate reward component
        self.f_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # h(s) network: learns a shaping (potential) function
        self.h_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state, action, next_state):
        # Concatenate state and action to compute f(s,a)
        sa = torch.cat([state, action], dim=-1)
        f_sa = self.f_net(sa)
        # Compute potential h(s) and h(s')
        h_s = self.h_net(state)
        h_next = self.h_net(next_state)
        # Compute the AIRL “logit” function: g(s,a,s') = f(s,a) + gamma * h(s') - h(s)
        g = f_sa + self.gamma * h_next - h_s
        return g
