import argparse
import os
import copy

import gym
import numpy as np
import rc_gym
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from ddpg import *
from networks import *
from experience import *

ENV = 'SSLGoToBall-v0'
PROCESSES_COUNT = 1
LEARNING_RATE = 0.0001
REPLAY_SIZE = 1500000
REPLAY_INITIAL = 256
BATCH_SIZE = 256
GAMMA = 0.95
REWARD_STEPS = 2
SAVE_FREQUENCY = 40000
STEP_GRAD_RATIO = 10


def get_env_specs(env_name):
    env = gym.make(env_name)
    return env.observation_space.shape[0], env.action_space.shape[0]


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"
    save_path = os.path.join("runs", "ddpg", args.name)
    os.makedirs(save_path, exist_ok=True)

    n_obs, n_acts = get_env_specs(ENV)

    act_net = DDPGActor(n_obs, n_acts).to(device)
    crt_net = DDPGCritic(n_obs, n_acts).to(device)

    # Playing
    act_net.share_memory()
    train_queue = mp.Queue(maxsize=BATCH_SIZE)
    finish_event = mp.Event()
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(
            target=data_func,
            args=(
                act_net,
                device,
                train_queue,
                finish_event,
                ENV,
                GAMMA,
                REWARD_STEPS
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    writer = SummaryWriter(save_path)
    tgt_act_net = TargetActor(act_net)
    tgt_crt_net = TargetCritic(crt_net)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)
    buffer = ExperienceReplayBuffer(buffer_size=REPLAY_SIZE)
    n_grads = 0
    n_samples = 0
    best_reward = None

    try:
        while True:
            for i in range(STEP_GRAD_RATIO):
                exp = train_queue.get()
                safe_exp = copy.deepcopy(exp)
                del(exp)
                if safe_exp is None:
                    raise Exception
                buffer.add(safe_exp)
                n_samples += 1

            if len(buffer) < REPLAY_INITIAL:
                continue

            batch = buffer.sample(BATCH_SIZE)
            states_v, actions_v, rewards_v, \
                dones_mask, last_states_v = \
                unpack_batch_ddpg(batch, device)

            # train critic
            crt_opt.zero_grad()
            q_v = crt_net(states_v, actions_v)
            last_act_v = tgt_act_net.target_model(
                last_states_v)
            q_last_v = tgt_crt_net.target_model(
                last_states_v, last_act_v)
            q_last_v[dones_mask] = 0.0
            q_ref_v = rewards_v.unsqueeze(dim=-1) + \
                q_last_v * (GAMMA**REWARD_STEPS)
            critic_loss_v = F.mse_loss(
                q_v, q_ref_v.detach())
            critic_loss_v.backward()
            crt_opt.step()

            # train actor
            act_opt.zero_grad()
            cur_actions_v = act_net(states_v)
            actor_loss_v = -crt_net(
                states_v, cur_actions_v)
            actor_loss_v = actor_loss_v.mean()
            actor_loss_v.backward()
            act_opt.step()

            tgt_act_net.sync(alpha=1 - 1e-3)
            tgt_crt_net.sync(alpha=1 - 1e-3)

            n_grads += 1

    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    finally:
        if train_queue:
            while train_queue.qsize() > 0:
                train_queue.get()

        print('queue is empty')

        print("Waiting for threads to finish...")
        for p in data_proc_list:
            p.terminate()
            p.join()

        del(train_queue)
        del(act_net)
