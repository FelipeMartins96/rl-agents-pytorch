import collections
import copy
import os
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
from agents.utils import HyperParameters, NStepTracer, generate_gif


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
