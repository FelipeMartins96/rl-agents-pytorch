from agents.utils.buffer import ReplayBuffer
import copy
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import gym
import numpy as np
import torch
from agents.ddpg.networks import TargetActor, TargetCritic
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
            info = {}
            ep_steps = 0
            ep_rw = [0]*hp.N_AGENTS
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                manager_obs = s[0]
                manager_action = trainer.manager_action(manager_obs)
                objectives = manager_action.reshape((-1, hp.OBJECTIVE_SIZE))
                workers_obs = trainer.workers_obs(
                    obs_env=s, objectives=objectives)
                workers_actions = trainer.workers_action(workers_obs)

                s_next, r, done, info = env.step(workers_actions)
                ep_steps += 1

                next_manager_obs = s[0]
                next_workers_obs = trainer.workers_obs(
                    obs_env=s_next, objectives=objectives)

                manager_rewards = trainer.manager_reward(r)
                workers_rewards = trainer.workers_rewards(
                    n_obs_env=s_next, objectives=objectives)

                obs = [manager_obs] + workers_obs
                actions = [manager_action] + workers_actions
                next_obs = [next_manager_obs] + next_workers_obs
                rewards = [manager_rewards] + workers_rewards

                for i in range(hp.N_AGENTS):
                    ep_rw[i] += rewards[i]

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
            info['ep_rw'] = ep_rw
            queue_m.put(info)


@dataclass
class FMHHP(HyperParameters):
    AGENT: str = "fmh_async"
    PERSIST_COMM: int = 8
    WORKER_OBS_IDX: list = None
    WORKER_REW_METHOD: function = lambda x, y: np.linalg.norm(x-y)
    MANAGER_REW_METHOD: function = np.sum


@dataclass
class FMHSACHP(FMHHP):
    AGENT: str = "fmhsac_async"
    ALPHA: float = None
    LOG_SIG_MAX: int = None
    LOG_SIG_MIN: int = None
    EPSILON: float = None
    OBJECTIVE_SIZE: int = None


class FMH:

    def __init__(self, methods: List, hp: FMHHP) -> None:

        self.manager = methods[0](hp)
        self.manager.gamma = 0.75

        self.workers = []
        for method in methods[1:]:
            self.workers.append(method(hp))

        self.hp = hp
        self.replay_buffers = []
        for _ in range(methods):
            buffer = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                                  observation_space=hp.observation_space,
                                  action_space=hp.action_space,
                                  device=hp.DEVICE
                                  )
            self.replay_buffers.append(buffer)

    def share_memory(self):
        self.manager.share_memory()
        [worker.share_memory() for worker in self.workers]

    def manager_reward(self, reward):
        return self.hp.MANAGER_REW_METHOD(reward)

    def workers_rewards(self, n_obs_env, objectives):
        indexes = self.hp.WORKER_OBS_IDX
        rew_function = self.hp.WORKER_REW_METHOD
        rewards = list()
        for next_obs, objective in zip(n_obs_env, objectives):
            reached_obj = next_obs[indexes]
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

    def manager_action(self, obs_manager):
        return self.manager.get_action(obs_manager)

    def workers_action(self, obs_workers):
        return [worker(obs) for worker, obs in zip(self.workers, obs_workers)]

    def experience(self, experiences):
        for buffer, exp in zip(self.replay_buffers, experiences):
            done = False
            if exp.last_state is not None:
                last_state = exp.last_state
            else:
                last_state = exp.state
                done = True
            buffer.add(
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
        agents = [self.manager] + self.workers
        for i, agent in enumerate(agents):
            self.save_agent(agent=agent, name=f'agent_{i}')

    def load(self, load_path):
        agents = [self.manager] + self.workers
        for i, agent in enumerate(agents):
            self.load_agent(agent=agent,
                            load_path=load_path,
                            name=f'agent_{i}')

    def update(self):
        pass
