import argparse
import copy
import datetime
import os
import time

import gym
import numpy as np
import rc_gym
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

import wandb
from agents.sac import (SACHP, GaussianPolicy, QNetwork, TargetCritic,
                        data_func, loss_sac)
from agents.utils import ReplayBuffer, save_checkpoint

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-e", "--env", required=True,
                        help="Name of the gym environment")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # Input Experiment Hyperparameters
    hp = SACHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=4,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=3,
        ALPHA=0.015,
        LOG_SIG_MAX=2,
        LOG_SIG_MIN=-20,
        EPSILON=1e-6,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=100000,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=100000,
        TOTAL_GRAD_STEPS=1000000,
        MULTI_AGENT=True
    )
    wandb.init(project='RoboCIn-RL', entity='goncamateus',
               name=hp.EXP_NAME, config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)

    # Actor-Critic
    pi = [GaussianPolicy(hp.N_OBS, hp.N_ACTS,
                        hp.LOG_SIG_MIN,
                        hp.LOG_SIG_MAX, hp.EPSILON).to(device)]*hp.N_AGENTS
    Q = [QNetwork(hp.N_OBS, hp.N_ACTS).to(device)]*hp.N_AGENTS
    # Entropy
    alpha = [hp.ALPHA]*hp.N_AGENTS
    target_entropy = -torch.prod(torch.Tensor(hp.N_ACTS).to(device)).item()
    log_alpha = [torch.zeros(1, requires_grad=True, device=device)]*hp.N_AGENTS

    # Playing
    [pi[i].share_memory() for i in range(hp.N_AGENTS)]
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
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
    tgt_Q = [TargetCritic(Q[i]) for i in range(hp.N_AGENTS)]
    pi_opt = [optim.Adam(pi[i].parameters(), lr=hp.LEARNING_RATE) for i in range(hp.N_AGENTS)]
    Q_opt = [optim.Adam(Q[i].parameters(), lr=hp.LEARNING_RATE) for i in range(hp.N_AGENTS)]
    alpha_optim = [optim.Adam([log_alpha[i]], lr=hp.LEARNING_RATE) for i in range(hp.N_AGENTS)]
    buffer = [ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE
                          )]*hp.N_AGENTS
    n_grads = 0
    n_samples = 0
    n_episodes = 0
    best_reward = None
    last_gif = None

    try:
        while n_grads < hp.TOTAL_GRAD_STEPS:
            metrics = {}
            ep_infos = list()
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
                    logs = {"ep_info/"+key: value for key,
                            value in safe_exp.items() if 'truncated' not in key}
                    ep_infos.append(logs)
                    n_episodes += 1
                else:
                    [buffer[i].add(
                        obs=safe_exp[i].state,
                        next_obs=safe_exp[i].last_state if safe_exp[i].last_state is not None else safe_exp[i].state,
                        action=safe_exp[i].action,
                        reward=safe_exp[i].reward,
                        done=False if safe_exp[i].last_state is not None else True
                    ) for i in range(hp.N_AGENTS)]
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()

            # Only start training after buffer is larger than initial value
            if buffer[0].size() < hp.REPLAY_INITIAL:
                continue
            
            for i in range(hp.N_AGENTS):
                # Sample a batch and load it as a tensor on device
                batch = buffer[i].sample(hp.BATCH_SIZE)
                pi_loss, Q_loss1, Q_loss2, log_pi = loss_sac(alpha[i],
                                                            hp.GAMMA,
                                                            batch, Q[i], pi[i],
                                                            tgt_Q[i], device)

                # train Entropy parameter

                alpha_loss = -(log_alpha[i] * (log_pi + target_entropy).detach())
                alpha_loss = alpha_loss.mean()

                alpha_optim[i].zero_grad()
                alpha_loss.backward()
                alpha_optim[i].step()

                alpha[i] = log_alpha[i].exp()
                metrics[f"agent_{i}/train/loss_alpha"] = alpha_loss.cpu().detach().numpy()
                metrics[f"agent_{i}/train/alpha"] = alpha[i].cpu().detach().numpy()

                # train actor - Maximize Q value received over every S
                pi_opt[i].zero_grad()
                pi_loss.backward()
                pi_opt[i].step()
                metrics[f"agent_{i}/train/loss_pi"] = pi_loss.cpu().detach().numpy()

                # train critic
                Q_loss = Q_loss1 + Q_loss2
                Q_opt[i].zero_grad()
                Q_loss.backward()
                Q_opt[i].step()

                metrics[f"agent_{i}/train/loss_Q1"] = Q_loss1.cpu().detach().numpy()
                metrics[f"agent_{i}/train/loss_Q2"] = Q_loss2.cpu().detach().numpy()

                # Sync target networks
                tgt_Q[i].sync(alpha=1 - 1e-3)

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics["counters/buffer_len"] = buffer[0].size()

            if ep_infos:
                for key in ep_infos[0].keys():
                    metrics[key] = np.mean([info[key] for info in ep_infos])

            # Log metrics
            wandb.log(metrics)
            if hp.SAVE_FREQUENCY and n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    hp=hp,
                    metrics={
                        'alpha': alpha,
                        'n_samples': n_samples,
                        'n_grads': n_grads,
                        'n_episodes': n_episodes
                    },
                    pi=pi,
                    Q=Q,
                    pi_opt=pi_opt,
                    Q_opt=Q_opt
                )

            if hp.GIF_FREQUENCY and n_grads % hp.GIF_FREQUENCY == 0:
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

        finish_event.set()
