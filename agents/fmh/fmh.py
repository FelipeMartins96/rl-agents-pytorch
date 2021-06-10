from agents.utils.noise import OrnsteinUhlenbeckNoise
import copy
import os
import time
from dataclasses import dataclass
from typing import List

import gym
import numpy as np
import torch
from agents.ddpg.networks import TargetActor, TargetCritic
from agents.utils.buffer import ReplayBuffer
from agents.utils.experience import ExperienceFirstLast
from agents.utils.experiment import HyperParameters
from agents.utils.gif import generate_gif


def data_func(
    trainer,
    queue_m,
    finish_event_m,
    sigma_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
    noise_manager = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=-1,
        max_value=1
    )
    noise_workers = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=-1,
        max_value=1
    )

    with torch.no_grad():
        while not finish_event_m.is_set():
            # Check for generate gif request
            gif_idx = -1
            with gif_req_m.get_lock():
                if gif_req_m.value != -1:
                    gif_idx = gif_req_m.value
                    gif_req_m.value = -1
            if gif_idx != -1:
                path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}.gif")
                generate_gif(env=env, filepath=path,
                             pi=copy.deepcopy(trainer), hp=hp)

            done = False
            s = env.reset()
            noise_manager.reset()
            noise_manager.sigma = sigma_m.value
            noise_workers.reset()
            noise_workers.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            ep_rw = [0]*hp.N_AGENTS
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                manager_obs = s[0]
                manager_action = trainer.manager_action(manager_obs)
                manager_action = noise_manager(manager_action)
                objectives = manager_action.reshape((-1, hp.OBJECTIVE_SIZE))
                workers_obs = trainer.workers_obs(obs_env=s,
                                                  objectives=objectives)
                workers_actions = trainer.workers_action(workers_obs,
                                                         noise_workers)
                s_next, r, done, info = env.step(workers_actions)
                ep_steps += 1

                next_manager_obs = s[0]
                next_workers_obs = trainer.workers_obs(obs_env=s_next,
                                                       objectives=objectives)

                manager_rewards = trainer.manager_reward(r)
                workers_rewards = trainer.workers_rewards(
                    n_obs_env=s_next, objectives=objectives
                )

                obs = [manager_obs] + workers_obs
                actions = [manager_action] + workers_actions
                next_obs = [next_manager_obs] + next_workers_obs
                rewards = [manager_rewards] + workers_rewards

                for i in range(hp.N_AGENTS-1):
                    ep_rw[i] += r[i]

                exp = list()
                for i in range(hp.N_AGENTS):
                    kwargs = {
                        'state': obs[i],
                        'action': actions[i],
                        'reward': rewards[i],
                        'last_state': next_obs[i]
                    }
                    exp.append(ExperienceFirstLast(**kwargs))
                queue_m.put(exp)
                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['ep_steps'] = ep_steps
            info['ep_rw'] = np.mean(ep_rw)
            info['noise'] = noise_manager.sigma
            queue_m.put(info)


@dataclass
class FMHHP(HyperParameters):
    AGENT: str = "fmh_async"
    PERSIST_COMM: int = 8
    WORKER_OBS_IDX: list = None
    OBJECTIVE_SIZE: int = None
    NOISE_SIGMA_INITIAL: float = None  # Initial action noise sigma
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None  # Action noise sigma decay
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: float = None  # Decay action noise every _ grad steps

    def MANAGER_REW_METHOD(self, x):
        return np.mean(x)

    def WORKER_REW_METHOD(self, x, y):
        return -np.linalg.norm(x-y)

    def __post_init__(self):
        super().__post_init__()
        self.WORKER_N_OBS = len(self.WORKER_OBS_IDX) + self.OBJECTIVE_SIZE


@dataclass
class FMHSACHP(FMHHP):
    AGENT: str = "fmhsac_async"
    ALPHA: float = None
    LOG_SIG_MAX: int = None
    LOG_SIG_MIN: int = None
    EPSILON: float = None


class FMH:

    last_manager_action = None
    action_idx = 0

    def __init__(self, methods: List, hp: FMHHP) -> None:

        hps = list()
        manager_hp = copy.deepcopy(hp)
        manager_hp.N_ACTS = (manager_hp.N_AGENTS-1)*manager_hp.OBJECTIVE_SIZE
        manager_hp.action_space.shape = (manager_hp.N_ACTS,)
        self.manager = methods[0](manager_hp)
        hps.append(manager_hp)

        worker_hp = copy.deepcopy(hp)
        worker_hp.N_OBS = hp.WORKER_N_OBS
        worker_hp.observation_space.shape = (worker_hp.N_OBS, )
        self.worker = methods[1](worker_hp)
        hps.append(worker_hp)

        self.hp = hp
        self.replay_buffers = []
        for hp in hps:
            buffer = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                                  observation_space=hp.observation_space,
                                  action_space=hp.action_space,
                                  device=hp.DEVICE
                                  )
            self.replay_buffers.append(buffer)
        self.update_index = 0

    def share_memory(self):
        self.manager.share_memory()
        self.worker.share_memory()

    def manager_reward(self, reward):
        return self.hp.MANAGER_REW_METHOD(reward)

    def workers_rewards(self, n_obs_env, objectives):
        indexes = self.hp.WORKER_OBS_IDX
        rew_function = self.hp.WORKER_REW_METHOD
        rewards = list()
        for next_obs, objective in zip(n_obs_env, objectives):
            reached_obj = next_obs[indexes[:self.hp.OBJECTIVE_SIZE]]
            rew = rew_function(reached_obj, objective)
            rewards.append(rew)
        return rewards

    def workers_obs(self, obs_env, objectives):
        indexes = self.hp.WORKER_OBS_IDX
        observations = list()
        for next_obs, objective in zip(obs_env, objectives):
            worker_obs = np.concatenate((next_obs[indexes], objective))
            observations.append(worker_obs)
        return observations

    def manager_action(self, obs_manager, train=True):
        if self.update_index < 20000 and train:
            persist_comm = 30
        else:
            persist_comm = self.hp.PERSIST_COMM
        if self.action_idx % persist_comm == 0 or not train:
            action = self.manager.get_action(obs_manager)
            if train:
                self.last_manager_action = action
                self.action_idx += 1
        else:
            action = self.last_manager_action
            self.action_idx += 1
        return action

    def workers_action(self, obs_workers, noise=lambda x: x):
        return noise(self.worker.get_action(obs_workers)).tolist()

    def experience(self, experiences):
        for i, exp in enumerate(experiences):
            done = False
            if exp.last_state is not None:
                last_state = exp.last_state
            else:
                last_state = exp.state
                done = True
            if i == 0:
                self.replay_buffers[0].add(
                    obs=exp.state,
                    next_obs=last_state,
                    action=exp.action,
                    reward=exp.reward,
                    done=done
                )
            else:
                self.replay_buffers[1].add(
                    obs=exp.state,
                    next_obs=last_state,
                    action=exp.action,
                    reward=exp.reward,
                    done=done
                )

    def save_agent(self, agent, name):
        torch.save(agent.pi.state_dict(),
                   f'{self.hp.CHECKPOINT_PATH}/{name}_actor.pth')
        torch.save(agent.Q.state_dict(),
                   f'{self.hp.CHECKPOINT_PATH}/{name}_critic.pth')
        torch.save(agent.pi_opt.state_dict(),
                   f'{self.hp.CHECKPOINT_PATH}/{name}_actor_optim.pth')
        torch.save(agent.Q_opt.state_dict(),
                   f'{self.hp.CHECKPOINT_PATH}/{name}_critic_optim.pth')

    def load_agent(self, agent, load_path, name):
        agent.pi.load_state_dict(torch.load(
            f'{load_path}/{name}_actor.pth'))
        agent.Q.load_state_dict(torch.load(
            f'{load_path}/{name}_critic.pth'))
        agent.pi_opt.load_state_dict(torch.load(
            f'{load_path}/{name}_actor_optim.pth'))
        agent.Q_opt.load_state_dict(torch.load(
            f'{load_path}/{name}_critic_optim.pth'))
        agent.tgt_pi = TargetActor(agent.pi)
        agent.tgt_Q = TargetCritic(agent.Q)

    def save(self):
        agents = [self.manager, self.worker]
        for i, agent in enumerate(agents):
            self.save_agent(agent=agent, name=f'agent_{i}')

    def load(self, load_path):
        agents = [self.manager, self.worker]
        for i, agent in enumerate(agents):
            self.load_agent(agent=agent,
                            load_path=load_path,
                            name=f'agent_{i}')

    def update(self):
        self.update_index += 1


class FMHSAC(FMH):

    def update(self):
        super().update()
        metrics = {}
        agents = [self.manager, self.worker]
        for i, agent in enumerate(agents):
            if self.update_index < 20000 and i == 0:
                continue
            batch = self.replay_buffers[i].sample(self.hp.BATCH_SIZE)
            loss = agent.update(batch)
            if loss:
                metrics.update({
                    f"agent_{i}/p_loss": loss[0],
                    f"agent_{i}/q1_loss": loss[1],
                    f"agent_{i}/q2_loss": loss[2],
                    f"agent_{i}/loss_alpha": loss[3],
                    f"agent_{i}/alpha": loss[4],
                    f"agent_{i}/mean(rew)": loss[5]
                })
        return metrics


class FMHDDPG(FMH):

    def update(self):
        super().update()
        metrics = {}
        agents = [self.manager, self.worker]
        for i, agent in enumerate(agents):
            if self.update_index < 20000 and i == 0:
                continue
            batch = self.replay_buffers[i].sample(self.hp.BATCH_SIZE)
            loss = agent.update(batch)
            if loss:
                metrics.update({
                    f"agent_{i}/p_loss": loss[0],
                    f"agent_{i}/q_loss": loss[1],
                    f"agent_{i}/mean(rew)": loss[2]
                })
        return metrics
