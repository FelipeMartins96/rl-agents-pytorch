import torch 
from torch.nn import functional as F
from torch import nn
import copy

class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_size)
        )

    def forward(self, x):
        return self.net(x)
    


class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size + act_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))


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
