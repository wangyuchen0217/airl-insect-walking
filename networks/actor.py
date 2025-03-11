import torch
import torch.nn as nn
from networks.utils import build_mlp, reparameterize, evaluate_lop_pi


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
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)
    
    
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
        return reparameterize(means, log_stds.clamp_(-20, 2))
