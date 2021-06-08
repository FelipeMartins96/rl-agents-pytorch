import argparse
import os

import gym
import numpy as np
import rsoccer_gym
import torch

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy
from agents.utils.gif import generate_gif


def get_env_specs(env_name):
    env = gym.make(env_name)
    return env.observation_space.shape[0], env.action_space.shape[0], env.spec.max_episode_steps


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

    generate_gif(env=env, filepath=args.checkpoint.replace(
        "pth", "gif").replace("checkpoint", "gif"), pi=pi, device=device)
