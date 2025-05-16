import torch
import torch.nn as nn
from networks.utils import build_mlp, reparameterize, evaluate_lop_pi, atanh


class ActorNetworkPolicy(nn.Module):
    """
    The actor network: maps states to actions.
    Uses a generic MLP to compute a policy mean and maintains a learnable log standard
    deviation (log_stds) for a Gaussian policy.
    """
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(), scale=1):
        super(ActorNetworkPolicy, self).__init__()
        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        # log_stds is a learnable parameter that defines the standard deviation for all actions. (torch.zeros(1, action_shape[0]) * 0.5)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0])*0.5)
        self.scale = scale # <--- add scale parameter

    def forward(self, states):
        """
        Compute the policy mean given states.
        A tanh activation is applied to bound the outputs.
        """
        mean = torch.tanh(self.net(states))
        return mean * self.scale # <--- apply scale parameter
    
    def sample(self, states):
        """
        Sample actions from the policy distribution using the reparameterization trick.
        Returns a tuple: (action, log probability).
        """
        """
        用 reparameterization 采样动作 + 修正后的 log_prob
        """
        mean = self.net(states)
        std = self.log_stds.exp().expand_as(mean)
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh) * self.scale

        # 修正 log_prob
        log_prob = -0.5 * ((noise ** 2) + 2 * self.log_stds + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = log_prob.sum(dim=-1)
        log_prob -= torch.log(1 - torch.tanh(pre_tanh).pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

    def evaluate_log_pi(self, states, actions):
        """
        Evaluate the log probabilities for given actions under the current policy.
        """
        """
        计算给定动作的 log_prob（包含 tanh 修正）
        """
        mean = self.net(states)
        std = self.log_stds.exp().expand_as(mean)

        # 反推 pre_tanh 的动作值
        squashed_action = actions / self.scale
        squashed_action = torch.clamp(squashed_action, -0.999, 0.999)
        pre_tanh = atanh(squashed_action)

        noise = (pre_tanh - mean) / std
        log_prob = -0.5 * ((noise ** 2) + 2 * self.log_stds + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = log_prob.sum(dim=-1)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6).sum(dim=-1)

        return log_prob
    
    
class StateDependentPolicy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))
