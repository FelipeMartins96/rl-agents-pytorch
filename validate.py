import gym
import rsoccer_gym
import torch
import torch.nn as nn
from agents.ddpg import DDPGActor
import numpy as np
import argparse

# Base on baselines implementation
class OrnsteinUhlenbeckAction(object):
    def __init__(self, theta=.17, dt=0.025, x0=None):
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        self.theta = theta
        self.mu = (action_space.high + action_space.low) / 2
        self.sigma = (action_space.high - self.mu) / 2
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class TeamSA:
    def __init__(self, path):
        self.pi = DDPGActor(40, 2)
        self.pi.load_state_dict(state_dict=torch.load(path)['pi_state_dict'])
        self.ou_actions = []
        for i in range(2):
            self.ou_actions.append(OrnsteinUhlenbeckAction())
    
    def __call__(self, obs):
        acts = np.zeros((3,2))
        acts[0] = self.pi(torch.Tensor(obs[0])).detach().numpy()
        acts[1] = self.ou_actions[0].sample()
        acts[2] = self.ou_actions[1].sample()

        return acts

    def reset(self):
        for ou in self.ou_actions:
            ou.reset()

class TeamIC:
    def __init__(self, path):
        self.pi = DDPGActor(40, 2)
        self.pi.load_state_dict(state_dict=torch.load(path)['pi_state_dict'])
    
    def __call__(self, obs):
        return self.pi(torch.Tensor(obs)).detach().numpy()

    def reset(self):
        pass

class TeamCC:
    def __init__(self, path):
        self.pi = DDPGActor(40, 6)
        self.pi.load_state_dict(state_dict=torch.load(path)['pi_state_dict'])
    
    def __call__(self, obs):
        return self.pi(torch.Tensor(obs[0])).detach().numpy().reshape(3,2)
        return acts

    def reset(self):
        pass

class TeamOU:
    def __init__(self):
        self.ou_actions = []
        for i in range(3):
            self.ou_actions.append(OrnsteinUhlenbeckAction())
    
    def __call__(self, obs):
        acts = np.zeros((3,2))
        acts[0] = self.ou_actions[0].sample()
        acts[1] = self.ou_actions[1].sample()
        acts[2] = self.ou_actions[2].sample()

        return acts

    def reset(self):
        for ou in self.ou_actions:
            ou.reset()

class TeamZero:
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return np.zeros((3,2))

    def reset(self):
        pass

# Using VSS Single Agent env
env = gym.make('VSSTNMT-v0')

BLUE_TEAMS = {
    'sa': TeamSA('sa.pth'),
    'rsa': TeamIC('sa.pth'),
    'il': TeamIC('il-ddpg.pth'),
    'jal': TeamCC('jal.pth'),
}

YELLOW_TEAMS = {
    'sa': TeamSA('sa.pth'),
    'rsa': TeamIC('sa.pth'),
    'il': TeamIC('il-ddpg.pth'),
    'jal': TeamCC('jal.pth'),
    'ou': TeamOU(),
    'zero': TeamZero(),
}

N_MATCHES = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    blue_name = args.name
    blue_team = BLUE_TEAMS[blue_name]
    # for blue_name, blue_team in BLUE_TEAMS.items():
    total_scores = []
    results = []
    for yellow_name, yellow_team in YELLOW_TEAMS.items():
        scores = []
        for _ in range(N_MATCHES):
            obs = env.reset()
            done = False
            while not done:
                # Step using random actions
                actions = np.zeros((6,2))
                actions[:3] = blue_team(obs[:3])
                actions[3:] = yellow_team(obs[3:])

                obs, reward, done, _ = env.step(actions)
                if done:
                    yellow_team.reset()
                    blue_team.reset()
                    obs = env.reset()
            scores.append(reward)
        results.append(f'\n{blue_name} vs {yellow_name}')
        results.append(f'Wins: {(np.array(scores) == 1).sum():04d}, Draws: {(np.array(scores) == 0).sum():04d}, Losses: {(np.array(scores) == -1).sum():04d},')
        results.append(f'Score: mean {np.mean(scores):.4f} std {np.std(scores):.4f}')
        total_scores += scores

    results.append(f'\n{blue_name} vs All')
    results.append(f'Wins: {(np.array(total_scores) == 1).sum():04d}, Draws: {(np.array(total_scores) == 0).sum():04d}, Losses: {(np.array(total_scores) == -1).sum():04d},')
    results.append(f'Score: mean {np.mean(total_scores):.4f} std {np.std(total_scores):.4f}')
        
    for l in results:
        print(l)