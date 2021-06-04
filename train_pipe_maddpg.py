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


import wandb
from agents.maddpg import MADDPGAgentTrainer, MADDPGHP
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


def rollout(
    trainers,
    queue_m,
    finish_event_m,
    gif_req_m,
    sigma_m,
    hp
):
    envs = MultiEnv(hp.ENV_NAME, hp.N_ROLLOUT_PROCESSES)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=-1,
        max_value=1
    )
    noise.reset()
    frames = []

    with torch.no_grad():
        # Check for generate gif request
        gif_idx = -1
        env_gif = -1

        s = envs.reset()
        ep_steps = np.array([0]*hp.N_ROLLOUT_PROCESSES)
        ep_rw = np.array([[0]*hp.N_AGENTS]*hp.N_ROLLOUT_PROCESSES, dtype=float)
        st_time = [time.perf_counter()]*hp.N_ROLLOUT_PROCESSES
        while not finish_event_m.is_set():
            # Step the environment
            a = []
            for i in range(hp.N_ROLLOUT_PROCESSES):
                env_act = [agent.action(obs, noise)
                           for agent, obs in zip(trainers, s[i])]
                a.append(env_act)

            s_next, r, done, info = envs.step(a)
            if gif_idx != -1:
                frame = envs.render(mode='rgb_array', env_idx=env_gif)
                frame = fromarray(frame)
                frame = frame.convert('P', palette=ADAPTIVE)
                frames.append(frame)

            ep_steps += 1
            ep_rw += r
            # Trace NStep rewards and add to mp queue
            for i in range(hp.N_ROLLOUT_PROCESSES):
                exp = list()
                for j in range(hp.N_AGENTS):
                    kwargs = {
                        'state': s[i][j],
                        'action': a[i][j],
                        'reward': r[i][j],
                        'last_state': s_next[i][j]
                    }
                    exp.append(ExperienceFirstLast(**kwargs))
                queue_m.put(exp)

                if done[i]:
                    if gif_idx != -1 and env_gif == i:
                        path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}.gif")
                        frames[0].save(
                            fp=path, 
                            format='GIF', 
                            append_images=frames[1:], 
                            save_all=True,
                            duration=25, 
                            loop=0
                        )
                        gif_idx = -1
                    with gif_req_m.get_lock():
                        if gif_req_m.value != -1:
                            env_gif = i
                            gif_idx = gif_req_m.value
                            gif_req_m.value = -1
                    s[i] = envs.reset(i)
                    info[i]['fps'] = ep_steps / \
                        (time.perf_counter() - st_time[i])
                    info[i]['ep_steps'] = ep_steps[i]
                    info[i]['ep_rw'] = np.mean(ep_rw[i])
                    info[i]['noise'] = noise.sigma
                    queue_m.put(info[i])
                    ep_steps[i] = 0
                    ep_rw[i] = 0
                    st_time[i] = time.perf_counter()
                    noise.reset()
                    noise.sigma = sigma_m.value
                    frames = []
                else:
                    s[i] = s_next[i]


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
        N_ROLLOUT_PROCESSES=3,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=1024,
        GAMMA=0.95,
        REWARD_STEPS=3,
        NOISE_SIGMA_INITIAL=0.1,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_MIN=0.15,
        NOISE_SIGMA_GRAD_STEPS=3000,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=1024,
        SAVE_FREQUENCY=10000,
        GIF_FREQUENCY=10000,
        TOTAL_GRAD_STEPS=1000000,
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
    data_proc = mp.Process(
        target=rollout,
        args=(
            trainers,
            exp_queue,
            finish_event,
            gif_req_m,
            sigma_m,
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
            for agent in trainers:
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
                    if isinstance(ep_infos[0][key], dict):
                        for i in range(hp.N_AGENTS):
                            for inner_key in ep_infos[0][key].keys():
                                metrics[f"ep_info/agent_{i}/{inner_key}"] = np.mean(
                                    [info[key][inner_key] for info in ep_infos])
                    else:
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
        data_proc.terminate()
        data_proc.join()

        del(exp_queue)
        del(trainers)

        finish_event.set()


if __name__ == "__main__":
    main(get_args())
