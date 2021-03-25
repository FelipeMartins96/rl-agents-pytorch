import argparse
import copy
import dataclasses
import datetime
import os
import time

import gym
import numpy as np
import rc_gym
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from agents.ddpg import (DDPGActor, DDPGCritic, HyperParameters, TargetActor,
                         TargetCritic, save_checkpoint, data_func)
from agents.utils import unpack_batch, ExperienceReplayBuffer, get_env_specs


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

    # Input Experiment Hyperparameters
    hp = HyperParameters(
        EXP_NAME=args.name,
        ENV_NAME='SSLGoToBall-v0',
        AGENT="ddpg_async",
        N_ROLLOUT_PROCESSES=1,
        LEARNING_RATE=0.0001,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=10000,
        EXP_GRAD_RATIO=10,
        SAVE_FREQUENCY=1000,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=2,
        NOISE_SIGMA_INITIAL=1.0,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_GRAD_STEPS=20000,
        GIF_FREQUENCY=20000
    )

    hp.SAVE_PATH = os.path.join("saves", hp.AGENT, hp.EXP_NAME)
    checkpoint_path = os.path.join(hp.SAVE_PATH, "Checkpoints")
    current_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    tb_path = os.path.join('runs',
                           hp.ENV_NAME + '_' + hp.EXP_NAME + '_' + current_time)
    os.makedirs(checkpoint_path, exist_ok=True)

    hp.N_OBS, hp.N_ACTS = get_env_specs(hp.ENV_NAME)

    pi = DDPGActor(hp.N_OBS, hp.N_ACTS).to(device)
    Q = DDPGCritic(hp.N_OBS, hp.N_ACTS).to(device)

    # Playing
    pi.share_memory()
    exp_queue = mp.Queue(maxsize=hp.BATCH_SIZE)
    finish_event = mp.Event()
    noise_sigma_m = mp.Value('f', hp.NOISE_SIGMA_INITIAL)
    gif_req_m = mp.Value('i', -1)
    data_proc_list = []
    for _ in range(hp.N_ROLLOUT_PROCESSES):
        data_proc = mp.Process(
            target=data_func,
            args=(
                pi,
                device,
                exp_queue,
                finish_event,
                noise_sigma_m,
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    writer = SummaryWriter(tb_path)
    tgt_pi = TargetActor(pi)
    tgt_Q = TargetCritic(Q)
    pi_opt = optim.Adam(pi.parameters(), lr=hp.LEARNING_RATE)
    Q_opt = optim.Adam(Q.parameters(), lr=hp.LEARNING_RATE)
    buffer = ExperienceReplayBuffer(buffer_size=hp.REPLAY_SIZE)
    n_grads = 0
    n_samples = 0
    n_episodes = 0
    best_reward = None

    # Record experiment parameters
    writer.add_text(
        tag="HyperParameters",
        text_string=str(hp).replace(',', "  \n"),
    )

    try:
        while True:
            metrics = {}
            st_time = time.perf_counter()
            # Collect EXP_GRAD_RATIO sample for each grad step
            new_samples = 0
            while new_samples < hp.EXP_GRAD_RATIO:
                exp = exp_queue.get()
                if exp is None:
                    raise Exception  # got None value in queue
                safe_exp = copy.deepcopy(exp)
                del(exp)

                # Dict is returned with end of episode info
                if isinstance(safe_exp, dict):
                    for key, value in safe_exp.items():
                        writer.add_scalar(
                            tag="ep_info/"+key,
                            scalar_value=value,
                            global_step=n_episodes
                        )
                    n_episodes += 1
                else:
                    buffer.add(safe_exp)
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()

            if len(buffer) < hp.REPLAY_SIZE:
                # Track buffer filling speed
                writer.add_scalar("buffer/len", len(buffer), n_samples)
                # Only start training after buffer is larger than initial value
                if len(buffer) < hp.REPLAY_INITIAL:
                    continue

            # Sample a batch and load it as a tensor on device
            batch = buffer.sample(hp.BATCH_SIZE)
            S_v, A_v, r_v, dones, S_next_v = unpack_batch(batch, device)

            # train critic
            Q_opt.zero_grad()
            Q_v = Q(S_v, A_v)  # expected Q for S,A
            A_next_v = tgt_pi(S_next_v)  # Get an Bootstrap Action for S_next
            Q_next_v = tgt_Q(S_next_v, A_next_v)  # Bootstrap Q_next
            Q_next_v[dones] = 0.0  # No bootstrap if transition is terminal
            # Calculate a reference Q value using the bootstrap Q
            Q_ref_v = r_v.unsqueeze(dim=-1) + Q_next_v * \
                (hp.GAMMA**hp.REWARD_STEPS)
            Q_loss_v = F.mse_loss(Q_v, Q_ref_v.detach())
            Q_loss_v.backward()
            Q_opt.step()
            metrics["train/loss_Q"] = Q_loss_v

            # train actor - Maximize Q value received over every S
            pi_opt.zero_grad()
            A_cur_v = pi(S_v)
            pi_loss_v = -Q(S_v, A_cur_v)
            pi_loss_v = pi_loss_v.mean()
            pi_loss_v.backward()
            pi_opt.step()
            metrics["train/loss_pi"] = pi_loss_v

            # Sync target networks
            tgt_pi.sync(alpha=1 - 1e-3)
            tgt_Q.sync(alpha=1 - 1e-3)

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)

            # Log metrics
            for key, value in metrics.items():
                writer.add_scalar(
                    tag=key,
                    scalar_value=value,
                    global_step=n_grads
                )

            if n_grads % hp.NOISE_SIGMA_GRAD_STEPS == 0:
                # This syntax is needed to be process-safe
                # The noise sigma value is accessed by the playing processes
                with noise_sigma_m.get_lock():
                    noise_sigma_m.value *= hp.NOISE_SIGMA_DECAY

            if n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    experiment=hp.EXP_NAME,
                    agent="ddpg_async",
                    pi=pi,
                    Q=Q,
                    pi_opt=pi_opt,
                    Q_opt=Q_opt,
                    noise_sigma=0,
                    n_samples=n_samples,
                    n_grads=n_grads,
                    n_episodes=n_episodes,
                    device=device,
                    checkpoint_path=checkpoint_path
                )

            if n_grads % hp.GIF_FREQUENCY == 0 and hp.GIF_FREQUENCY != 0:
                gif_req_m.value = n_grads

    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    finally:
        if exp_queue:
            while exp_queue.qsize() > 0:
                exp_queue.get()

        print('queue is empty')

        print("Waiting for threads to finish...")
        for p in data_proc_list:
            p.terminate()
            p.join()

        del(exp_queue)
        del(pi)
