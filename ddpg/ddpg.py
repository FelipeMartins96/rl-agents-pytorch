import torch
import collections
import time
import gym
import numpy as np
from experience import *
from noise import *
import os
from dataclasses import dataclass

@dataclass
class HyperParameters:
    """Class containing all experiment hyperparameters"""
    EXP_NAME: str
    ENV_NAME: str
    N_ROLLOUT_PROCESSES: int
    LEARNING_RATE: float
    REPLAY_SIZE: int # Maximum Replay Buffer Sizer
    REPLAY_INITIAL: int # Minimum experience buffer size to start training
    EXP_GRAD_RATIO: int  # Number of collected experiences for every grad step
    SAVE_FREQUENCY: int  # Save checkpoint every _ grad_steps
    BATCH_SIZE: int
    GAMMA: float # Reward Decay
    REWARD_STEPS: float  # For N-Steps Tracing
    NOISE_SIGMA_INITIAL: float # Initial action noise sigma
    NOISE_THETA: float
    NOISE_SIGMA_DECAY: float  # Action noise sigma decay
    NOISE_SIGMA_GRAD_STEPS: int  # Decay action noise every _ grad steps
    N_OBS: int = 0
    N_ACTS: int = 0

def unpack_batch_ddpg(
    batch,
    device="cpu"
):
    '''From a batch of experience, return values in Tensor form on device'''
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = torch.Tensor(states).to(device)
    actions_v = torch.Tensor(actions).to(device)
    rewards_v = torch.Tensor(rewards).to(device)
    last_states_v = torch.Tensor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


def data_func(
    pi,
    device,
    queue_m,
    finish_event_m,
    env_name,
    gamma,
    reward_steps,
    sigma_m,
    theta
):
    env = gym.make(env_name)
    tracer = NStepTracer(n=reward_steps, gamma=gamma)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value, 
        theta=theta, 
        min_value=env.action_space.low,
        max_value=env.action_space.high
    )
    
    with torch.no_grad():
        while not finish_event_m.is_set():
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
