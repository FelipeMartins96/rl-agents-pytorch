import collections
import copy
import os
import time
from dataclasses import dataclass

import gym
import torch
import torch.nn.functional as F
from agents.sac import GaussianPolicy, QNetwork, TargetCritic
from agents.utils import (ExperienceFirstLast, HyperParameters, NStepTracer,
                          generate_gif)
from torch.optim import Adam


@dataclass
class SACHP(HyperParameters):
    ALPHA: float = None
    LOG_SIG_MAX: int = None
    LOG_SIG_MIN: int = None
    EPSILON: float = None
    AGENT: str = "sac_async"


def data_func(
    pi,
    device,
    queue_m,
    finish_event_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
    if hp.MULTI_AGENT:
        tracer = [NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)]*hp.N_AGENTS
    else:
        tracer = NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)

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
                             pi=copy.deepcopy(pi), hp=hp)

            done = False
            s = env.reset()
            if hp.MULTI_AGENT:
                [tracer[i].reset() for i in range(hp.N_AGENTS)]
            info = {}
            ep_steps = 0
            if hp.MULTI_AGENT:
                ep_rw = [0]*hp.N_AGENTS
            else:
                ep_rw = 0
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a = pi.get_action(s_v)
                s_next, r, done, info = env.step(a)

                ep_steps += 1
                if hp.MULTI_AGENT:
                    for i in range(hp.N_AGENTS):
                        ep_rw[i] += r[f'robot_{i}']
                else:
                    ep_rw += r
                # Trace NStep rewards and add to mp queue
                if hp.MULTI_AGENT:
                    exp = list()
                    for i in range(hp.N_AGENTS):
                        kwargs = {
                            'state': s[i],
                            'action': a[i],
                            'reward': r[f'robot_{i}'],
                            'last_state': s_next[i]
                        }
                        exp.append(ExperienceFirstLast(**kwargs))
                    queue_m.put(exp)
                else:
                    tracer.add(s, a, r, done)
                    while tracer:
                        queue_m.put(tracer.pop())

                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['ep_steps'] = ep_steps
            info['ep_rw'] = ep_rw
            queue_m.put(info)


def loss_sac(alpha, gamma, batch, crt_net, act_net,
             tgt_crt_net):

    state_batch = batch.observations
    action_batch = batch.actions
    reward_batch = batch.rewards
    mask_batch = batch.dones.bool()
    next_state_batch = batch.next_observations

    with torch.no_grad():
        next_state_action, next_state_log_pi, _ = act_net.sample(
            next_state_batch
        )
        qf1_next_target, qf2_next_target = tgt_crt_net.target_model(
            next_state_batch, next_state_action
        )
        min_qf_next_target = (
            torch.min(qf1_next_target, qf2_next_target)
            - alpha * next_state_log_pi
        )
        min_qf_next_target[mask_batch] = 0.0
        next_q_value = reward_batch + gamma * min_qf_next_target

    # Two Q-functions to mitigate

    # positive bias in the policy improvement step
    qf1, qf2 = crt_net(state_batch, action_batch)

    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf1_loss = F.mse_loss(qf1, next_q_value)

    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value)

    pi, log_pi, _ = act_net.sample(state_batch)

    qf1_pi, qf2_pi = crt_net(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    policy_loss = alpha * log_pi
    policy_loss = policy_loss - min_qf_pi
    policy_loss = policy_loss.mean()

    return policy_loss, qf1_loss, qf2_loss, log_pi


class SAC:

    def __init__(self, hp):
        device = hp.DEVICE
        # Actor-Critic
        self.pi = GaussianPolicy(hp.N_OBS, hp.N_ACTS,
                                 hp.LOG_SIG_MIN,
                                 hp.LOG_SIG_MAX, hp.EPSILON).to(device)
        self.Q = QNetwork(hp.N_OBS, hp.N_ACTS).to(device)
        # Entropy
        self.alpha = hp.ALPHA
        self.target_entropy = - \
            torch.prod(torch.Tensor(hp.N_ACTS).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # Training
        self.tgt_Q = TargetCritic(self.Q)
        self.pi_opt = Adam(self.pi.parameters(), lr=hp.LEARNING_RATE)
        self.Q_opt = Adam(self.Q.parameters(), lr=hp.LEARNING_RATE)
        self.alpha_optim = Adam([self.log_alpha], lr=hp.LEARNING_RATE)

        self.gamma = hp.GAMMA**hp.REWARD_STEPS
        self.hp = hp
    
    def share_memory(self):
        self.pi.share_memory()
        self.Q.share_memory()

    def loss(self, batch):
        state_batch = batch.observations
        action_batch = batch.actions
        reward_batch = batch.rewards
        mask_batch = batch.dones.bool()
        next_state_batch = batch.next_observations

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.pi.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.tgt_Q.target_model(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            min_qf_next_target[mask_batch] = 0.0
            next_q_value = reward_batch + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate

        # positive bias in the policy improvement step
        qf1, qf2 = self.Q(state_batch, action_batch)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.pi.sample(state_batch)

        qf1_pi, qf2_pi = self.Q(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        policy_loss = policy_loss.mean()

        return policy_loss, qf1_loss, qf2_loss, log_pi

    def update(self, batch, metrics):
        pi_loss, Q_loss1, Q_loss2, log_pi = self.loss(batch)
        # train Entropy parameter

        alpha_loss = -(self.log_alpha * (log_pi +
                       self.target_entropy).detach())
        alpha_loss = alpha_loss.mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        alpha = self.log_alpha.exp()
        alpha_loss = alpha_loss.cpu().detach().numpy()
        metrics["train/alpha"] = alpha.cpu().detach().numpy()

        # train actor - Maximize Q value received over every S
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        pi_loss = pi_loss.cpu().detach().numpy()

        # train critic
        Q_loss = Q_loss1 + Q_loss2
        self.Q_opt.zero_grad()
        Q_loss.backward()
        self.Q_opt.step()

        Q_loss1 = Q_loss1.cpu().detach().numpy()
        Q_loss2 = Q_loss2.cpu().detach().numpy()

        # Sync target networks
        self.tgt_Q.sync(alpha=1 - 1e-3)

        return pi_loss, Q_loss1, Q_loss2, alpha_loss, alpha.cpu().detach().numpy()
