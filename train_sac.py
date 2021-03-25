import argparse
import copy
import datetime
from math import gamma
import os
import time

import gym
import rc_gym
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from agents.sac import SACHP, data_func, save_checkpoint, loss_sac,\
    GaussianPolicy, QNetwork, TargetCritic
from agents.utils import ExperienceReplayBuffer, get_env_specs


if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-e", "--env", default='SSLGoToBall-v0',
                        help="Environment Id")
    parser.add_argument("-p", "--num_processes", default=1,
                        help="NUmber of rollout processes", type=int)
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # Input Experiment Hyperparameters
    hp = SACHP(
        EXP_NAME=args.name,
        ENV_NAME=args.env,
        AGENT="sac_async",
        N_ROLLOUT_PROCESSES=args.num_processes,
        LEARNING_RATE=0.0001,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=10000,
        EXP_GRAD_RATIO=10,
        SAVE_FREQUENCY=1000,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=2,
        GIF_FREQUENCY=20000
    )

    hp.SAVE_PATH = os.path.join("saves", hp.AGENT, hp.EXP_NAME)
    checkpoint_path = os.path.join(hp.SAVE_PATH, "Checkpoints")
    current_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    tb_path = os.path.join('runs',
                           hp.ENV_NAME + '_' + hp.EXP_NAME + '_' + current_time)
    os.makedirs(checkpoint_path, exist_ok=True)

    hp.N_OBS, hp.N_ACTS = get_env_specs(hp.ENV_NAME)

    # Actor-Critic
    pi = GaussianPolicy(hp.N_OBS, hp.N_ACTS,
                        hp.LOG_SIG_MIN,
                        hp.LOG_SIG_MAX, hp.EPSILON).to(device)
    Q = QNetwork(hp.N_OBS, hp.N_ACTS).to(device)
    # Entropy
    alpha = hp.ALPHA
    target_entropy = -torch.prod(torch.Tensor(hp.N_ACTS).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    # Playing
    pi.share_memory()
    exp_queue = mp.Queue(maxsize=hp.BATCH_SIZE)
    finish_event = mp.Event()
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
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    writer = SummaryWriter(tb_path)
    tgt_Q = TargetCritic(Q)
    pi_opt = optim.Adam(pi.parameters(), lr=hp.LEARNING_RATE)
    Q_opt = optim.Adam(Q.parameters(), lr=hp.LEARNING_RATE)
    alpha_optim = optim.Adam([log_alpha], lr=hp.LEARNING_RATE)
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
            pi_loss, Q_loss1, Q_loss2, log_pi = loss_sac(alpha, hp.GAMMA,
                                                         batch, Q, pi,
                                                         tgt_Q, device)

            # train Entropy parameter

            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach())
            alpha_loss = alpha_loss.mean()

            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            alpha = log_alpha.exp()
            alpha_tlogs = alpha.clone()
            metrics["train/loss_alpha"] = alpha_loss
            metrics["train/alpha"] = alpha

            # train actor - Maximize Q value received over every S
            pi_opt.zero_grad()
            pi_loss.backward()
            pi_opt.step()
            metrics["train/loss_pi"] = pi_loss

            # train critic
            Q_loss = Q_loss1 + Q_loss2
            Q_opt.zero_grad()
            Q_loss.backward()
            Q_opt.step()

            metrics["train/loss_Q1"] = Q_loss1
            metrics["train/loss_Q2"] = Q_loss2

            # Sync target networks
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

            if n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    experiment=hp.EXP_NAME,
                    agent=hp.AGENT,
                    pi=pi,
                    Q=Q,
                    pi_opt=pi_opt,
                    Q_opt=Q_opt,
                    alpha=alpha,
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
