import argparse
import os
import copy
import datetime
import dataclasses
import time

import gym
import numpy as np
import rc_gym
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy


def get_env_specs(env_name):
    env = gym.make(env_name)
    return env.observation_space.shape[0], env.action_space.shape[0], env.spec.max_episode_steps


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="checkpoint to load")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    checkpoint = torch.load(args.path)

    env = gym.make(checkpoint['ENV_NAME'])

    if checkpoint['AGENT'] == 'ddpg_async':
        pi = DDPGActor(checkpoint['N_OBS'], checkpoint['N_ACTS']).to(device)
    elif checkpoint['AGENT'] == 'sac_async':
        pi = GaussianPolicy(checkpoint['N_OBS'], checkpoint['N_ACTS'],
                        checkpoint['LOG_SIG_MIN'],
                        checkpoint['LOG_SIG_MAX'], checkpoint['EPSILON']).to(device)
    else:
        raise AssertionError

    pi.load_state_dict(checkpoint['pi_state_dict'])
    pi.eval()
    
    while True:
        done = False
        s = env.reset()
        info = {}
        ep_steps = 0
        ep_rw = 0
        st_time = time.perf_counter()
        for i in range(checkpoint['MAX_EPISODE_STEPS']):
            # Step the environment
            s_v = torch.Tensor(s).to(device)
            a = pi.get_action(s_v)
            s_next, r, done, info = env.step(a)
            ep_steps += 1
            ep_rw += r
            env.render()
            if done:
                break

            # Set state for next step
            s = s_next

        info['fps'] = ep_steps / (time.perf_counter() - st_time)
        info['ep_steps'] = ep_steps
        info['ep_rw'] = ep_rw
        print(info)
