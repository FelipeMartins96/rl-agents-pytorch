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

ENV = 'Pendulum-v0'
ROLLOUT_PROCESSES_COUNT = 1
LEARNING_RATE = 0.0001
REPLAY_SIZE = 1500000 # Maximum Replay Buffer Sizer
REPLAY_INITIAL = 256 # Minimum experience buffer size to start training
BATCH_SIZE = 256
GAMMA = 0.95 # Reward Decay
REWARD_STEPS = 2 # For N-Steps Tracing
SAVE_FREQUENCY = 1000 # Save checkpoint every _ grad_steps
EXP_GRAD_RATIO = 5 # Number of collected experiences for every grad step
NOISE_SIGMA_INITIAL = 1.0 # Initial action noise sigma
NOISE_SIGMA_DECAY = 0.999 # Action noise sigma decay 
NOISE_SIGMA_GRAD_STEPS = 1000 # Decay action noise every _ grad steps
NOISE_THETA = 0.15


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
    
    path = os.path.join("saves", "ddpg", args.name)
    checkpoint_path = os.path.join(path, "Checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    n_obs, n_acts = get_env_specs(ENV)

    pi = DDPGActor(n_obs, n_acts).to(device)
    Q = DDPGCritic(n_obs, n_acts).to(device)

    # Playing
    pi.share_memory()
    exp_queue = mp.Queue(maxsize=BATCH_SIZE)
    finish_event = mp.Event()
    noise_sigma_m = mp.Value('f', NOISE_SIGMA_INITIAL)
    data_proc_list = []
    for _ in range(ROLLOUT_PROCESSES_COUNT):
        data_proc = mp.Process(
            target=data_func,
            args=(
                pi,
                device,
                exp_queue,
                finish_event,
                ENV,
                GAMMA,
                REWARD_STEPS,
                noise_sigma_m,
                NOISE_THETA
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    writer = SummaryWriter()
    tgt_pi = TargetActor(pi)
    tgt_Q = TargetCritic(Q)
    pi_opt = optim.Adam(pi.parameters(), lr=LEARNING_RATE)
    Q_opt = optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    buffer = ExperienceReplayBuffer(buffer_size=REPLAY_SIZE)
    n_grads = 0
    n_samples = 0
    best_reward = None

    try:
        while True:
            for i in range(EXP_GRAD_RATIO):
                exp = exp_queue.get()
                if exp is None:
                    raise Exception #got None value in queue
                safe_exp = copy.deepcopy(exp)
                del(exp)
                buffer.add(safe_exp)
                n_samples += 1

            if len(buffer) < REPLAY_INITIAL:
                continue

            batch = buffer.sample(BATCH_SIZE)
            S_v, A_v, r_v, dones, S_next_v = unpack_batch_ddpg(batch, device)

            # train critic
            Q_opt.zero_grad()
            Q_v = Q(S_v, A_v)
            A_next_v = tgt_pi(S_next_v)
            Q_next_v = tgt_Q(S_next_v, A_next_v)
            Q_next_v[dones] = 0.0
            q_ref_v = r_v.unsqueeze(dim=-1) + Q_next_v * (GAMMA**REWARD_STEPS)
            critic_loss_v = F.mse_loss(Q_v, q_ref_v.detach())
            critic_loss_v.backward()
            Q_opt.step()

            # train actor
            pi_opt.zero_grad()
            A_cur_v = pi(S_v)
            actor_loss_v = -Q(S_v, A_cur_v)
            actor_loss_v = actor_loss_v.mean()
            actor_loss_v.backward()
            pi_opt.step()

            tgt_pi.sync(alpha=1 - 1e-3)
            tgt_Q.sync(alpha=1 - 1e-3)

            n_grads += 1
            if n_grads % NOISE_SIGMA_GRAD_STEPS == 0:
                # This syntax is needed to be process-safe
                # The noise sigma value is accessed by the playing processes
                with noise_sigma_m.get_lock():
                    noise_sigma_m.value *= NOISE_SIGMA_DECAY
            
            if n_grads % SAVE_FREQUENCY == 0:
                print("actor_loss = {}, critic_loss = {}".format(critic_loss_v, actor_loss_v))
                save_checkpoint(
                    experiment=args.name,
                    agent="ddpg_async",
                    pi=pi,
                    Q=Q,
                    pi_opt=pi_opt,
                    Q_opt=Q_opt,
                    noise_sigma=0,
                    n_samples=n_samples,
                    n_grads=n_grads,
                    device=device,
                    checkpoint_path=checkpoint_path
                )

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
