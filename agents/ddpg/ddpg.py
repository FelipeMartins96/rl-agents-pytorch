import torch
import collections
import time
import gym
import copy
import numpy as np
from agents.utils import NStepTracer, OrnsteinUhlenbeckNoise, generate_gif, HyperParameters
import os
from dataclasses import dataclass

@dataclass
class DDPGHP(HyperParameters):
    AGENT: str= "ddpg_async"
    NOISE_SIGMA_INITIAL: float= None # Initial action noise sigma
    NOISE_THETA: float= None
    NOISE_SIGMA_DECAY: float= None  # Action noise sigma decay
    NOISE_SIGMA_GRAD_STEPS: float= None  # Decay action noise every _ grad steps

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
