import torch
import numpy as np
import PIL
import os


def generate_gif(
    env,
    filepath,
    pi,
    hp,
    max_episode_steps=1200,
    resize_to=None,
    duration=25
):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment
    filepath : str
    pi : nn.Module
    max_episode_steps : int
    resize_to : tuple of ints, optional
    duration : float, optional
    """

    # collect frames
    MA_METHODS = ['maddpg_async', 'fmh_async']
    frames = []
    s = env.reset()
    for t in range(max_episode_steps):

        if hp.AGENT not in MA_METHODS:
            s_v = torch.Tensor(s).to(hp.DEVICE)
            a = pi.get_action(s_v)
            s_next, r, done, info = env.step(a)

        elif hp.AGENT == "maddpg_async":
            a = [agent.action(obs) for agent, obs in zip(pi, s)]
            s_next, r, done, info = env.step(a)

        else:
            manager_obs = s[0]
            manager_action = pi.manager_action(manager_obs)
            objectives = manager_action.reshape((-1, hp.OBJECTIVE_SIZE))
            workers_obs = pi.workers_obs(obs_env=s, objectives=objectives)
            workers_actions = pi.workers_action(workers_obs)
            s_next, r, done, info = env.step(workers_actions)

        # store frame
        frame = env.render(mode='rgb_array')
        frame = PIL.Image.fromarray(frame)
        frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render(mode='rgb_array')
    frame = PIL.Image.fromarray(frame)
    frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    frames[0].save(
        fp=filepath,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
