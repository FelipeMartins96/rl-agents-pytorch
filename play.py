import argparse
import os
import time

import gym
import numpy as np
import rsoccer_gym
import torch

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="checkpoint to load")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    checkpoint = torch.load(args.checkpoint)

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
