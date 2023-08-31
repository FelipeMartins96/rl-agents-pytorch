import gym
import rsoccer_gym
import torch
import torch.nn as nn
from agents.ddpg import DDPGActor
import numpy as np
import argparse
import PIL

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
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.pi = DDPGActor(state_dict['N_OBS'], 2)
        self.pi.load_state_dict(state_dict=state_dict['pi_state_dict'])
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
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.pi = DDPGActor(state_dict['N_OBS'], 2)
        self.pi.load_state_dict(state_dict=state_dict['pi_state_dict'])
    
    def __call__(self, obs):
        return self.pi(torch.Tensor(obs)).detach().numpy()

    def reset(self):
        pass

class TeamCC:
    def __init__(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.pi = DDPGActor(state_dict['N_OBS'], 6)
        self.pi.load_state_dict(state_dict=state_dict['pi_state_dict'])
    
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


BLUE_TEAMS = {
    'sa-0': TeamSA('nets/sa-0.pth'),
    'rsa-0': TeamIC('nets/sa-0.pth'),
    'il-ddpg-0': TeamIC('nets/il-ddpg-0.pth'),
    'il-maddpg-0': TeamIC('nets/il-maddpg-0.pth'),
    'jal-0': TeamCC('nets/jal-0.pth'),
    'sa-1': TeamSA('nets/sa-1.pth'),
    'rsa-1': TeamIC('nets/sa-1.pth'),
    'il-ddpg-1': TeamIC('nets/il-ddpg-1.pth'),
    'il-maddpg-1': TeamIC('nets/il-maddpg-1.pth'),
    'jal-1': TeamCC('nets/jal-1.pth'),
    'sa-2': TeamSA('nets/sa-2.pth'),
    'rsa-2': TeamIC('nets/sa-2.pth'),
    'il-ddpg-2': TeamIC('nets/il-ddpg-2.pth'),
    'il-maddpg-2': TeamIC('nets/il-maddpg-2.pth'),
    'jal-2': TeamCC('nets/jal-2.pth'),
}

YELLOW_TEAMS = [
    {
        'sa-0': TeamSA('nets/sa-0.pth'),
        'sa-1': TeamSA('nets/sa-1.pth'),
        'sa-2': TeamSA('nets/sa-2.pth'),
    },{
        'rsa-0': TeamIC('nets/sa-0.pth'),
        'rsa-1': TeamIC('nets/sa-1.pth'),
        'rsa-2': TeamIC('nets/sa-2.pth'),
    },{
        'il-ddpg-0': TeamIC('nets/il-ddpg-0.pth'),
        'il-ddpg-1': TeamIC('nets/il-ddpg-1.pth'),
        'il-ddpg-2': TeamIC('nets/il-ddpg-2.pth'),
    },{
        'il-maddpg-0': TeamIC('nets/il-maddpg-0.pth'),
        'il-maddpg-1': TeamIC('nets/il-maddpg-1.pth'),
        'il-maddpg-2': TeamIC('nets/il-maddpg-2.pth'),
    },{
        'jal-0': TeamCC('nets/jal-0.pth'),
        'jal-1': TeamCC('nets/jal-1.pth'),
        'jal-2': TeamCC('nets/jal-2.pth'),
    },{
        'ou-99': TeamOU(),
        'zero-99': TeamZero(),
    }
]

N_MATCHES = 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-p", "--pool", required=True, help="pool")
    args = parser.parse_args()


    # Using VSS Single Agent env
    env = gym.make('VSSTNMT-v0')
    blue_name = args.name
    blue_team = BLUE_TEAMS[blue_name]
    # for blue_name, blue_team in BLUE_TEAMS.items():
    total_scores = []
    results = []
    for yellow_name, yellow_team in YELLOW_TEAMS[int(args.pool)].items():
        scores = []
        steps = []
        for _ in range(N_MATCHES):
            frames = []
            obs = env.reset()
            done = False
            step = 0
            while not done:
                # Step using random actions
                actions = np.zeros((6,2))
                actions[:3] = blue_team(obs[:3])
                actions[3:] = yellow_team(obs[3:])

                frame = env.render(mode='rgb_array')
                frame = PIL.Image.fromarray(frame)
                frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
                frames.append(frame)
                obs, reward, done, _ = env.step(actions)
                step += 1
                if done:
                    yellow_team.reset()
                    blue_team.reset()
                    obs = env.reset()


                    # store last frame
                    frame = env.render(mode='rgb_array')
                    frame = PIL.Image.fromarray(frame)
                    frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
                    frames.append(frame)

                    import os
                    if reward == 1:
                        filepath = f'videos/wins/{len(os.listdir("videos/wins"))}.webp'
                    elif reward == 0:
                        filepath = f'videos/draws/{len(os.listdir("videos/draws"))}.webp'
                    else:
                        filepath = f'videos/losses/{len(os.listdir("videos/losses"))}.webp'
        
                    # generate gif
                    frames[0].save(
                        fp=filepath, 
                        format='webp', 
                        append_images=frames[1:], 
                        save_all=True,
                        duration=25, 
                        loop=0
                    )

            steps.append(step)
            scores.append(reward)
        results.append(f'{blue_name.split("-")[0]},{yellow_name.split("-")[0]},{blue_name.split("-")[1]},{yellow_name.split("-")[1]},{(np.array(scores) == 1).sum():04d},{(np.array(scores) == 0).sum():04d},{(np.array(scores) == -1).sum():04d},{np.mean(steps):.3f},{np.std(steps):.3f}')
        total_scores += scores
    
    print('#start-results#')
    for l in results:
        print(l)