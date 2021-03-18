import torch
import collections
import gym
import numpy as np
from experience import *


def unpack_batch_ddpg(
    batch,
    device="cpu"
):
    '''From a batch of experience, return values in Tensor form on device'''
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = torch.Tensor(states).to(device)
    actions_v = torch.Tensor(actions).to(device)
    rewards_v = torch.Tensor(rewards).to(device)
    last_states_v = torch.Tensor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


def data_func(
    pi,
    device,
    queue,
    finish_event,
    env_name,
    gamma,
    reward_steps
):

    env = gym.make(env_name)
    tracer = NStepTracer(n=reward_steps, gamma=gamma)

    with torch.no_grad():
        while not finish_event.is_set():
            done = False
            s = env.reset()

            while not done:
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a_v = pi(s_v)
                a = a_v.cpu().numpy()
                a = np.clip(a, -1, 1)
                s_next, r, done, info = env.step(a)
                env.render()
                # Trace NStep rewards and add to mp queue
                tracer.add(s, a, r, done)
                while tracer:
                    queue.put(tracer.pop())

                # Set state for next step
                s = s_next
