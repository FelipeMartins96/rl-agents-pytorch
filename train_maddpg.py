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
import PIL
from PIL.Image import fromarray, ADAPTIVE
import pprint


import wandb
from agents.maddpg import MADDPGAgentTrainer, MADDPGHP, data_func
from agents.utils import (ExperienceFirstLast, MultiEnv, OrnsteinUhlenbeckNoise,
                          ReplayBuffer, generate_gif, gif, save_checkpoint)


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


def get_trainers(hp):
    trainers = []
    trainer = MADDPGAgentTrainer
    for i in range(hp.N_AGENTS):
        trainers.append(trainer(i, hp))

    return trainers


def main(args):
    device = "cuda" if args.cuda else "cpu"
    mp.set_start_method('spawn')
    # Input Experiment Hyperparameters
    hp = MADDPGHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=4,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=1024,
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
        MULTI_AGENT=True,
        DISCRETE=False
    )
    wandb.init(project='RoboCIn-RL', name=hp.EXP_NAME,
               entity='robocin', config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)
    # Training
    trainers = get_trainers(hp)

    # Playing
    [trainers[i].pi.share_memory() for i in range(hp.N_AGENTS)]
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    gif_req_m = mp.Value('i', -1)
    sigma_m = mp.Value('f', hp.NOISE_SIGMA_INITIAL)
    data_proc_list = list()
    for _ in range(hp.N_ROLLOUT_PROCESSES):
        data_proc = mp.Process(
            target=data_func,
            args=(
                trainers,
                exp_queue,
                finish_event,
                sigma_m,
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

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
                    for i, exp in enumerate(safe_exp):
                        if exp.last_state is not None:
                            last_state = exp.last_state
                        else:
                            last_state = exp.state
                        trainers[i].experience(exp.state, exp.action,
                                               exp.reward, last_state,
                                               False if exp.last_state is not None else True)
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()

            # Only start training after buffer is larger than initial value
            if len(trainers[i].replay_buffer) < hp.REPLAY_INITIAL:
                continue

            # Sample a batch and load it as a tensor on device
            for agent in trainers:
                agent.preupdate()
            for i, agent in enumerate(trainers):
                loss = agent.update(trainers)
                if loss:
                    metrics.update({
                        "{}/q_loss".format(agent.name): loss[0],
                        "{}/p_loss".format(agent.name): loss[1],
                        "{}/mean(target_q)".format(agent.name): loss[2],
                        "{}/mean(rew)".format(agent.name): loss[3],
                        "{}/mean(target_q_next)".format(agent.name): loss[4],
                        "{}/std(target_q)".format(agent.name): loss[5]
                    })
                if ep_infos:
                    info = ep_infos[0]
                    info_metrics = {}
                    for key, value in info[f'ep_info/robot_{i}'].items():
                        info_metrics[f'{agent.name}/{key}'] = value
                    metrics.update(info_metrics)

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics["counters/buffer_len"] = len(trainers[i].replay_buffer)

            if ep_infos:
                for key in ep_infos[0].keys():
                    if not isinstance(ep_infos[0][key], dict):
                        metrics[key] = np.mean([info[key]
                                               for info in ep_infos])

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
                        'n_samples': n_samples,
                        'n_grads': n_grads,
                        'n_episodes': n_episodes
                    },
                    pi=[trainers[i].pi for i in range(hp.N_AGENTS)],
                    Q=[trainers[i].Q for i in range(hp.N_AGENTS)],
                    pi_opt=[trainers[i].pi_opt for i in range(hp.N_AGENTS)],
                    Q_opt=[trainers[i].Q_opt for i in range(hp.N_AGENTS)]
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
        del(trainers)

        finish_event.set()


if __name__ == "__main__":
    main(get_args())
