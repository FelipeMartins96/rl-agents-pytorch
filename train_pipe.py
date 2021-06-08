import argparse
import copy
import datetime
import os
import time

import gym
import numpy as np
import rsoccer_gym
import torch
import torch.multiprocessing as mp
import wandb

from agents.sac import SAC, SACHP
from agents.utils import (NStepTracer, ReplayBuffer, generate_gif, gif,
                          save_checkpoint, MultiEnv)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-e", "--env", required=True,
                        help="Name of the gym environment")
    args = parser.parse_args()
    return args


def rollout(
    agent,
    device,
    queue_m,
    finish_event_m,
    gif_req_m,
    hp
):
    envs = MultiEnv(hp.ENV_NAME, hp.N_ROLLOUT_PROCESSES)
    tracer = [NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)
              ]*hp.N_ROLLOUT_PROCESSES

    with torch.no_grad():
        # Check for generate gif request
        gif_idx = -1

        s = envs.reset()
        ep_steps = np.array([0]*hp.N_ROLLOUT_PROCESSES)
        ep_rw = np.array([0]*hp.N_ROLLOUT_PROCESSES, dtype=float)
        st_time = [time.perf_counter()]*hp.N_ROLLOUT_PROCESSES
        while not finish_event_m.is_set():
            # Step the environment
            s_v = torch.Tensor(s).to(device)
            a = agent.pi.get_action(s_v)
            s_next, r, done, info = envs.step(a)
            if gif_idx != -1:
                envs.render(mode='rgb_array', env_idx=gif_idx)

            ep_steps += 1
            ep_rw += r
            # Trace NStep rewards and add to mp queue
            for i in range(hp.N_ROLLOUT_PROCESSES):
                tracer[i].add(s[i], a[i], r[i], done[i])
                while tracer[i]:
                    queue_m.put(tracer[i].pop())

                if done[i]:
                    if gif_idx != -1:
                        gif_idx = -1
                    with gif_req_m.get_lock():
                        if gif_req_m.value != -1:
                            gif_idx = i
                            gif_req_m.value = -1
                    s[i] = envs.reset(i)
                    info[i]['fps'] = ep_steps / (time.perf_counter() - st_time[i])
                    info[i]['ep_steps'] = ep_steps[i]
                    info[i]['ep_rw'] = ep_rw[i]
                    queue_m.put(info[i])
                    ep_steps[i] = 0
                    ep_rw[i] = 0
                    st_time[i] = time.perf_counter()
                else:
                    s[i] = s_next[i]


def main(args):
    device = "cuda" if args.cuda else "cpu"
    mp.set_start_method('spawn')
    # Input Experiment Hyperparameters
    hp = SACHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=3,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=3,
        ALPHA=0.015,
        LOG_SIG_MAX=2,
        LOG_SIG_MIN=-20,
        EPSILON=1e-6,
        REPLAY_SIZE=100000,
        REPLAY_INITIAL=512,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=100000,
        TOTAL_GRAD_STEPS=1000000
    )
    wandb.init(project='RoboCIn-RL', name=hp.EXP_NAME,
               entity='robocin', config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)
    # Training
    sac = SAC(hp)
    buffer = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE
                          )

    # Playing
    sac.share_memory()
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    gif_req_m = mp.Value('i', -1)
    data_proc = mp.Process(
        target=rollout,
        args=(
            sac,
            device,
            exp_queue,
            finish_event,
            gif_req_m,
            hp
        )
    )
    data_proc.start()

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
                    if safe_exp.last_state is not None:
                        last_state = safe_exp.last_state
                    else:
                        last_state = safe_exp.state
                    buffer.add(
                        obs=safe_exp.state,
                        next_obs=last_state,
                        action=safe_exp.action,
                        reward=safe_exp.reward,
                        done=False if safe_exp.last_state is not None else True
                    )
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()

            # Only start training after buffer is larger than initial value
            if buffer.size() < hp.REPLAY_INITIAL:
                continue

            # Sample a batch and load it as a tensor on device
            batch = buffer.sample(hp.BATCH_SIZE)
            metrics["train/loss_pi"], metrics["train/loss_Q1"], \
                metrics["train/loss_Q2"], metrics["train/loss_alpha"], \
                metrics["train/alpha"] = sac.update(batch=batch,
                                                    metrics=metrics)

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
                    metrics[key] = np.mean([info[key] for info in ep_infos])

            # Log metrics
            wandb.log(metrics)
            if hp.SAVE_FREQUENCY and n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    hp=hp,
                    metrics={
                        'alpha': sac.alpha,
                        'n_samples': n_samples,
                        'n_grads': n_grads,
                        'n_episodes': n_episodes
                    },
                    pi=sac.pi,
                    Q=sac.Q,
                    pi_opt=sac.pi_opt,
                    Q_opt=sac.Q_opt
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
        data_proc.terminate()
        data_proc.join()

        del(exp_queue)
        del(sac)

        finish_event.set()


if __name__ == "__main__":
    main(get_args())
