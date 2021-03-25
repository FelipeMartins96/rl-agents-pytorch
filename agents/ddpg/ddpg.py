import torch
import collections
import time
import gym
import copy
import numpy as np
from agents.utils.experience import NStepTracer
from agents.utils.noise import OrnsteinUhlenbeckNoise
from agents.utils.gif import generate_gif
import os
from dataclasses import dataclass


@dataclass
class HyperParameters:
    """Class containing all experiment hyperparameters"""
    EXP_NAME: str
    ENV_NAME: str
    AGENT: str
    N_ROLLOUT_PROCESSES: int
    LEARNING_RATE: float
    REPLAY_SIZE: int  # Maximum Replay Buffer Sizer
    REPLAY_INITIAL: int  # Minimum experience buffer size to start training
    EXP_GRAD_RATIO: int  # Number of collected experiences for every grad step
    SAVE_FREQUENCY: int  # Save checkpoint every _ grad_steps
    BATCH_SIZE: int
    GAMMA: float  # Reward Decay
    REWARD_STEPS: float  # For N-Steps Tracing
    NOISE_SIGMA_INITIAL: float  # Initial action noise sigma
    NOISE_THETA: float
    NOISE_SIGMA_DECAY: float  # Action noise sigma decay
    NOISE_SIGMA_GRAD_STEPS: int  # Decay action noise every _ grad steps
    GIF_FREQUENCY: int = -1
    N_OBS: int = 0
    N_ACTS: int = 0
    SAVE_PATH: str = ""


def data_func(
    pi,
    device,
    queue_m,
    finish_event_m,
    sigma_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
    tracer = NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low,
        max_value=env.action_space.high
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
                generate_gif(env=env, filepath=os.path.join(hp.SAVE_PATH,\
                    f"gifs/{gif_idx:09d}.gif"), pi=copy.deepcopy(pi), 
                    max_episode_steps=1000, device=device)
            
            
            done = False
            s = env.reset()
            noise.reset()
            noise.sigma = sigma_m.value
            ep_steps = 0
            ep_rw = 0
            st_time = time.perf_counter()
            while not done:
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a_v = pi(s_v)
                a = a_v.cpu().numpy()
                a = noise(a)
                s_next, r, done, info = env.step(a)
                ep_steps += 1
                ep_rw += r

                # Trace NStep rewards and add to mp queue
                tracer.add(s, a, r, done)
                while tracer:
                    queue_m.put(tracer.pop())

                if done:
                    info['fps'] = ep_steps / (time.perf_counter() - st_time)
                    info['noise'] = noise.sigma
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
    noise_sigma,
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
        "n_samples": n_samples,
        "n_grads": n_grads,
        "n_episodes": n_episodes,
        "device": device
    }
    filename = os.path.join(
        checkpoint_path, "checkpoint_{:09}.pth".format(n_grads))
    torch.save(checkpoint, filename)
