import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state.clone(), action.clone()], 1)
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions,
                 log_sig_min, log_sig_max,
                 epsilon, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x2 = F.relu(self.linear2(x1))
        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=self.log_sig_min,
                              max=self.log_sig_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_action = torch.log(
            self.action_scale * (1 - y_t.pow(2)) + self.epsilon
        )
        log_prob = log_prob - log_action
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, state):
        action, _, _ = self.sample(state.unsqueeze(0))
        return action.detach().cpu().numpy()[0]

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class TargetActor(TargetNet):
    def __call__(self, S):
        return self.target_model(S)


class TargetCritic(TargetNet):
    def __call__(self, S, A):
        return self.target_model(S, A)

