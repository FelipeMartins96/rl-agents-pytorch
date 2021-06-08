import argparse
import copy
import dataclasses
import datetime
import os
import time

import gym
import numpy as np
import rsoccer_gym
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

import wandb
from agents.ddpg import (DDPGHP, DDPGActor, DDPGCritic, TargetActor,
                         TargetCritic, data_func)
from agents.utils import ReplayBuffer, save_checkpoint, unpack_batch, ExperienceFirstLast

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
    hp = DDPGHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=3,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=3,
        NOISE_SIGMA_INITIAL=0.8,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_MIN=0.15,
        NOISE_SIGMA_GRAD_STEPS=3000,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=100000,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=10000,
        TOTAL_GRAD_STEPS=2000000,
        MULTI_AGENT=True
    )
    wandb.init(project='RoboCIn-RL', name=hp.EXP_NAME, config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)

    pi = DDPGActor(hp.N_OBS, hp.N_ACTS).to(device)
    Q = DDPGCritic(hp.N_OBS, hp.N_ACTS).to(device)

    # Playing
    pi.share_memory()
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    sigma_m = mp.Value('f', hp.NOISE_SIGMA_INITIAL)
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
                sigma_m,
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    tgt_pi = TargetActor(pi)
    tgt_Q = TargetCritic(Q)
    pi_opt = optim.Adam(pi.parameters(), lr=hp.LEARNING_RATE)
    Q_opt = optim.Adam(Q.parameters(), lr=hp.LEARNING_RATE)
    buffer = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE
                          )
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
                    for exp in safe_exp:
                        if exp.last_state is not None:
                            last_state = exp.last_state
                        else:
                            last_state = exp.state
                        buffer.add(
                            obs=exp.state,
                            next_obs=last_state,
                            action=exp.action,
                            reward=exp.reward,
                            done=False if exp.last_state is not None else True
                        )
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()


            # Only start training after buffer is larger than initial value
            if buffer.size() < hp.REPLAY_INITIAL:
                continue

            # Sample a batch and load it as a tensor on device
            batch = buffer.sample(hp.BATCH_SIZE)
            S_v = batch.observations
            A_v = batch.actions
            r_v = batch.rewards
            dones = batch.dones
            S_next_v = batch.next_observations

            # train critic
            Q_opt.zero_grad()
            Q_v = Q(S_v, A_v)  # expected Q for S,A
            A_next_v = tgt_pi(S_next_v)  # Get an Bootstrap Action for S_next
            Q_next_v = tgt_Q(S_next_v, A_next_v)  # Bootstrap Q_next
            Q_next_v[dones == 1.] = 0.0  # No bootstrap if transition is terminal
            # Calculate a reference Q value using the bootstrap Q
            Q_ref_v = r_v + Q_next_v * (hp.GAMMA**hp.REWARD_STEPS)
            Q_loss_v = F.mse_loss(Q_v, Q_ref_v.detach())
            Q_loss_v.backward()
            Q_opt.step()
            metrics["train/loss_Q"] = Q_loss_v.cpu().detach().numpy()

            # train actor - Maximize Q value received over every S
            pi_opt.zero_grad()
            A_cur_v = pi(S_v)
            pi_loss_v = -Q(S_v, A_cur_v)
            pi_loss_v = pi_loss_v.mean()
            pi_loss_v.backward()
            pi_opt.step()
            metrics["train/loss_pi"] = pi_loss_v.cpu().detach().numpy()

            # Sync target networks
            tgt_pi.sync(alpha=1 - 1e-3)
            tgt_Q.sync(alpha=1 - 1e-3)

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics["counters/buffer_len"] = buffer.size()

            if ep_infos:
                for key in ep_infos[0].keys():
                    if isinstance(ep_infos[0][key], dict):
                        for i in range(hp.N_AGENTS):
                            for inner_key in ep_infos[0][key].keys():
                                metrics[f"ep_info/agent_{i}/{inner_key}"] = np.mean(
                                    [info[key][inner_key] for info in ep_infos])
                    else:
                        metrics[key] = np.mean([info[key] for info in ep_infos])

            # Log metrics
            wandb.log(metrics)

            if hp.NOISE_SIGMA_DECAY and sigma_m.value > hp.NOISE_SIGMA_MIN \
                and n_grads % hp.NOISE_SIGMA_GRAD_STEPS == 0:
                # This syntax is needed to be process-safe
                # The noise sigma value is accessed by the playing processes
                with sigma_m.get_lock():
                    sigma_m.value *= hp.NOISE_SIGMA_DECAY

            if hp.SAVE_FREQUENCY and n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    hp=hp,
                    metrics={
                        'noise_sigma': sigma_m.value,
                        'n_samples': n_samples,
                        'n_episodes': n_episodes,   
                        'n_grads': n_grads,
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
