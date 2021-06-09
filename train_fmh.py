import argparse
import copy
import datetime
import os
import time

import gym
import numpy as np
import rsoccer_gym
import torch.multiprocessing as mp

import wandb
from agents.fmh import FMH, FMHSACHP, data_func
from agents.sac.sac import SAC
from agents.utils import gif

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
    parser.add_argument("-o", "--obs_idx", required=True,
                        help="Name of the gym environment")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # Input Experiment Hyperparameters
    hp = FMHSACHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        WORKER_OBS_IDX=[int(idx) for idx in args.obs_idx.split(',')], 
        OBJECTIVE_SIZE=2,
        N_ROLLOUT_PROCESSES=2,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=1024,
        GAMMA=0.95,
        REWARD_STEPS=3,
        ALPHA=0.015,
        LOG_SIG_MAX=2,
        LOG_SIG_MIN=-20,
        EPSILON=1e-6,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=1024,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=10000,
        TOTAL_GRAD_STEPS=2000000,
        MULTI_AGENT=True,
    )
    wandb.init(project='RoboCIn-RL', name=hp.EXP_NAME, entity='robocin', config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)

    # Method instace
    hp.N_AGENTS += 1
    fmh = FMH(methods=[SAC]*hp.N_AGENTS, hp=hp)
    # Playing
    fmh.share_memory()
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    gif_req_m = mp.Value('i', -1)
    data_proc_list = []
    for _ in range(hp.N_ROLLOUT_PROCESSES):
        data_proc = mp.Process(
            target=data_func,
            args=(
                fmh,
                exp_queue,
                finish_event,
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    n_grads = 0
    n_samples = 0
    n_episodes = 0
    best_reward = None
    last_gif = 0

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
                    fmh.experience(safe_exp)
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()

            # Only start training after buffer is larger than initial value
            if fmh.replay_buffers[0].size() < hp.REPLAY_INITIAL:
                continue

            # Update networks and log metrics
            training_metrics = fmh.update()
            metrics.update(training_metrics)
            for i in range(hp.N_AGENTS-1):
                if ep_infos:
                    info = ep_infos[0]
                    info_metrics = {}
                    for key, value in info[f'ep_info/robot_{i}'].items():
                        info_metrics[f'agent_{i}/{key}'] = value
                    metrics.update(info_metrics)

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics["counters/buffer_len"] = fmh.replay_buffers[0].size()

            if ep_infos:
                for key in ep_infos[0].keys():
                    if not isinstance(ep_infos[0][key], dict):
                        metrics[key] = np.mean([info[key]
                                               for info in ep_infos])

            # Check if there is a new gif
            gifs = [int(file.split('.')[0]) for file in os.listdir(hp.GIF_PATH)]
            gifs.sort()
            if gifs and gifs[-1] > last_gif:
                last_gif = gifs[-1]
                path = os.path.join(hp.GIF_PATH, f"{last_gif:09d}.gif")
                metrics.update({"gif": wandb.Video(path, fps=10, format="gif")})
            # Log metrics
            wandb.log(metrics)
            if hp.SAVE_FREQUENCY and n_grads % hp.SAVE_FREQUENCY == 0:
                fmh.save()

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
        del(fmh)

        finish_event.set()
