import collections
import copy
import os
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
from agents.maddpg.buffer import ReplayBuffer
from agents.maddpg.networks import Actor, Critic, TargetActor, TargetCritic
from agents.utils import (ExperienceFirstLast, HyperParameters, NStepTracer,
                          OrnsteinUhlenbeckNoise, generate_gif)
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


@dataclass
class MADDPGHP(HyperParameters):
    AGENT: str = "maddpg_async"
    NOISE_SIGMA_INITIAL: float = None  # Initial action noise sigma
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None  # Action noise sigma decay
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: float = None  # Decay action noise every _ grad steps
    DISCRETE: float = None


def data_func(
    trainers,
    queue_m,
    finish_event_m,
    sigma_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low[0],
        max_value=env.action_space.high[1]
    )

    with torch.no_grad():
        while not finish_event_m.is_set():
            # Check for generate gif request
            gif_idx = -1
            with gif_req_m.get_lock():
                if gif_req_m.value != -1:
                    gif_idx = gif_req_m.value
                    gif_req_m.value = -1
            if gif_idx != -1:
                path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}.gif")
                generate_gif(env=env, filepath=path,
                             pi=copy.deepcopy(trainers), hp=hp)

            done = False
            s = env.reset()
            noise.reset()
            noise.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            ep_rw = [0]*hp.N_AGENTS
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                actions = [agent.action(obs) for agent, obs in zip(trainers, s)]
                a = [noise(act) for act in actions]
                s_next, r, done, info = env.step(a)
                ep_steps += 1
                if hp.MULTI_AGENT:
                    for i in range(hp.N_AGENTS):
                        ep_rw[i] += r[i]
                else:
                    ep_rw += r

                # Trace NStep rewards and add to mp queue
                exp = list()
                for i in range(hp.N_AGENTS):
                    kwargs = {
                        'state': s[i],
                        'action': a[i],
                        'reward': r[i],
                        'last_state': s_next[i]
                    }
                    exp.append(ExperienceFirstLast(**kwargs))
                queue_m.put(exp)
                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['noise'] = noise.sigma
            info['ep_steps'] = ep_steps
            info['ep_rw'] = ep_rw
            queue_m.put(info)


def onehot_from_logits(logits):
    """
    Given batch of logits, return one-hot sample 
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class MADDPGAgentTrainer(object):
    def __init__(self, agent_index, args):
        self.name = f'agent_{agent_index}'
        self.n = args.N_AGENTS
        self.agent_index = agent_index
        self.args = args
        self.discrete = args.DISCRETE
        obs_shape_n = args.N_OBS
        act_shape_n = args.N_ACTS

        # Train stuff
        self.pi = Actor(obs_shape_n, act_shape_n).to(args.DEVICE)
        self.Q = Critic(obs_shape_n*self.n, act_shape_n*self.n).to(args.DEVICE)
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_Q = TargetCritic(self.Q)
        self.pi_opt = Adam(self.pi.parameters(), lr=args.LEARNING_RATE)
        self.Q_opt = Adam(self.Q.parameters(), lr=args.LEARNING_RATE)
        self.grad_clipping = 0.5

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.min_replay_buffer_len = args.REPLAY_INITIAL
        self.replay_sample_index = None

    def save(self):
        torch.save(self.pi.state_dict(),
                   f'{self.args.CHECKPOINT_PATH}/{self.name}_actor.pth')
        torch.save(self.Q.state_dict(),
                   f'{self.args.CHECKPOINT_PATH}/{self.name}_critic.pth')
        torch.save(self.pi_opt.state_dict(),
                   f'{self.args.CHECKPOINT_PATH}/{self.name}_actor_optim.pth')
        torch.save(self.Q_opt.state_dict(),
                   f'{self.args.CHECKPOINT_PATH}/{self.name}_critic_optim.pth')

    def load(self, load_path):
        self.pi.load_state_dict(torch.load(
            f'{load_path}/{self.name}_actor.pth'))
        self.Q.load_state_dict(torch.load(
            f'{load_path}/{self.name}_critic.pth'))
        self.pi_opt.load_state_dict(torch.load(
            f'{load_path}/{self.name}_actor_optim.pth'))
        self.Q_opt.load_state_dict(torch.load(
            f'{load_path}/{self.name}_critic_optim.pth'))
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_Q = TargetCritic(self.Q)

    def action(self, obs, noise=lambda x: x):
        obs = torch.Tensor([obs]).to(self.args.DEVICE)
        act = self.pi(obs)
        if self.discrete:
            act = onehot_from_logits(act).detach().cpu().numpy().squeeze()
        else:
            act = torch.tanh(act).detach().cpu().numpy().squeeze()
            act = noise(act)
        return act

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents):
        metrics = {}

        if len(self.replay_buffer) < self.min_replay_buffer_len:  # replay buffer is not large enough
            return

        self.replay_sample_index = self.replay_buffer.make_index(
            self.args.BATCH_SIZE)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        new_acts_n = []
        target_act_next_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(
                index)
            obs_v = torch.Tensor(obs).to(self.args.DEVICE).float()
            obs_next_v = torch.Tensor(obs_next).to(self.args.DEVICE).float()
            act_v = torch.Tensor(act).to(self.args.DEVICE).float()
            obs_n.append(obs_v)
            obs_next_n.append(obs_next_v)
            act_n.append(act_v)
            new_act = agents[i].pi(obs_v)
            if i == self.agent_index:
                my_act_logits = new_act
                if self.discrete:
                    new_act = gumbel_softmax(new_act, hard=True)
                else:
                    new_act = torch.tanh(new_act)
            else:
                if self.discrete:
                    new_act = onehot_from_logits(new_act)
                else:
                    new_act = torch.tanh(new_act)
            new_acts_n.append(new_act)
            tgt_res = agents[i].tgt_pi(obs_next_v)
            if self.discrete:
                tgt_res = onehot_from_logits(tgt_res)
            else:
                tgt_res = torch.tanh(tgt_res)
            target_act_next_n.append(tgt_res)
        _, _, rew, _, done = self.replay_buffer.sample_index(index)
        rew_v = torch.Tensor(rew).to(self.args.DEVICE).float().unsqueeze(1)
        dones_v = torch.Tensor(done).to(self.args.DEVICE).float()

        obs_nv = torch.cat(obs_n, dim=1).to(self.args.DEVICE)
        obs_next_nv = torch.cat(obs_next_n, dim=1).to(self.args.DEVICE)
        act_nv = torch.cat(act_n, dim=1).to(self.args.DEVICE)
        new_acts_nv = torch.cat(new_acts_n, dim=1).to(self.args.DEVICE)
        target_act_next_nv = torch.cat(
            target_act_next_n, dim=1).to(self.args.DEVICE)

        # train critic
        Q_v = self.Q(obs_nv, act_nv)  # expected Q for S,A
        # Get an Bootstrap Action for S_next
        Q_next_v = self.tgt_Q(
            obs_next_nv, target_act_next_nv)  # Bootstrap Q_next
        Q_next_v[dones_v == 1.] = 0.0  # No bootstrap if transition is terminal
        # Calculate a reference Q value using the bootstrap Q
        Q_ref_v = rew_v + Q_next_v * self.args.GAMMA
        self.Q_opt.zero_grad()
        Q_loss_v = torch.mean(torch.square(Q_ref_v.detach() - Q_v))
        Q_loss_v.backward()
        clip_grad_norm_(self.Q.parameters(), 0.5)
        self.Q_opt.step()
        metrics["train/loss_Q"] = Q_loss_v.cpu().detach().numpy()

        # train actor - Maximize Q value received over every S
        self.pi_opt.zero_grad()
        pi_loss_v = -self.Q(obs_nv, new_acts_nv)
        pi_loss_v = pi_loss_v.mean()
        pi_loss_v += (my_act_logits**2).mean() * 1e-3
        pi_loss_v.backward()
        clip_grad_norm_(self.pi.parameters(), 0.5)
        self.pi_opt.step()
        metrics["train/loss_pi"] = pi_loss_v.cpu().detach().numpy()

        # Sync target networks
        self.tgt_pi.sync(alpha=0.99)
        self.tgt_Q.sync(alpha=0.99)

        return [metrics["train/loss_Q"], metrics["train/loss_pi"], np.mean(Q_v.cpu().detach().numpy()), np.mean(rew), np.mean(Q_next_v.cpu().detach().numpy()), np.std(Q_v.cpu().detach().numpy())]
