import argparse
import os

import gym
import numpy as np
import rsoccer_gym
import torch
import wandb

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy


def get_env_specs(env_name):
    env = gym.make(env_name)
    return (
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.spec.max_episode_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable cuda"
    )
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-c", "--checkpoint", required=True, help="checkpoint to load")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"
    wandb.init(project="reward_alphas", name=args.name, entity="robocin")

    checkpoint = torch.load(args.checkpoint)

    env = gym.make('VSSStrat-v0')

    if checkpoint["AGENT"] == "ddpg_async":
        pi = DDPGActor(checkpoint["N_OBS"], checkpoint["N_ACTS"]).to(device)
    elif checkpoint["AGENT"] == "sac_async":
        pi = GaussianPolicy(
            checkpoint["N_OBS"],
            checkpoint["N_ACTS"],
            checkpoint["LOG_SIG_MIN"],
            checkpoint["LOG_SIG_MAX"],
            checkpoint["EPSILON"],
        ).to(device)
    else:
        raise AssertionError

    pi.load_state_dict(checkpoint["pi_state_dict"])
    pi.eval()

    for i in range(1000):
        s = env.reset()
        done = False
        while not done:
            s_v = torch.Tensor(s).to(device)
            a = pi.get_action(s_v)
            s_next, r, done, info = env.step(a)
            # env.render()
            s = s_next
        logs = {
            "ep_info/" + key: value for key, value in info.items() if "truncated" not in key
        }
        wandb.log(logs)
    env.close()

