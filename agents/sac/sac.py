import collections
import copy
import os
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn.functional as F

from agents.utils import HyperParameters, NStepTracer, generate_gif, unpack_batch


@dataclass
class SACHP(HyperParameters):
    ALPHA: float = 0.015
    LOG_SIG_MAX: int = 2
    LOG_SIG_MIN: int = -20
    EPSILON: float= 1e-6


def data_func(
    pi,
    device,
    queue_m,
    finish_event_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
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
                path = os.path.join(hp.SAVE_PATH, f"gifs/{gif_idx:09d}.gif")
                generate_gif(env=env, filepath=path,
                             pi=copy.deepcopy(pi),
                             max_episode_steps=1000, device=device)

            done = False
            s = env.reset()
            ep_steps = 0
            ep_rw = 0
            st_time = time.perf_counter()
            while not done:
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a = pi.get_action(s_v)
                s_next, r, done, info = env.step(a)
                ep_steps += 1
                ep_rw += r

                # Trace NStep rewards and add to mp queue
                tracer.add(s, a, r, done)
                while tracer:
                    queue_m.put(tracer.pop())

                if done:
                    info['fps'] = ep_steps / (time.perf_counter() - st_time)
                    info['ep_steps'] = ep_steps
                    info['ep_rw'] = ep_rw
                    queue_m.put(info)

                # Set state for next step
                s = s_next


def loss_sac(alpha, gamma, batch, crt_net, act_net,
             tgt_crt_net, device):

    state_batch, action_batch, reward_batch,\
        mask_batch, next_state_batch = unpack_batch(batch, device)

    reward_batch = reward_batch.unsqueeze_(1)

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


def save_checkpoint(
    experiment: str,
    agent: str,
    pi,
    Q,
    pi_opt,
    Q_opt,
    alpha,
    n_samples,
    n_grads,
    n_episodes,
    device,
    checkpoint_path: str
):
    checkpoint = {
        "name": experiment,
        "agent": agent,
        "pi_state_dict": pi.state_dict(),
        "Q_state_dict": Q.state_dict(),
        "pi_opt_state_dict": pi_opt.state_dict(),
        "Q_opt_state_dict": Q_opt.state_dict(),
        "alpha": alpha,
        "n_samples": n_samples,
        "n_grads": n_grads,
        "n_episodes": n_episodes,
        "device": device
    }
    filename = os.path.join(
        checkpoint_path, "checkpoint_{:09}.pth".format(n_grads))
    torch.save(checkpoint, filename)
