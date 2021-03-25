from dataclasses import dataclass

import gym
import rc_gym
import torch
import os


@dataclass
class HyperParameters:
    """Class containing all experiment hyperparameters"""
    EXP_NAME: str
    ENV_NAME: str
    N_ROLLOUT_PROCESSES: int
    LEARNING_RATE: float
    REPLAY_SIZE: int  # Maximum Replay Buffer Sizer
    REPLAY_INITIAL: int  # Minimum experience buffer size to start training
    EXP_GRAD_RATIO: int  # Number of collected experiences for every grad step
    SAVE_FREQUENCY: int  # Save checkpoint every _ grad_steps
    BATCH_SIZE: int
    GAMMA: float  # Reward Decay
    REWARD_STEPS: float  # For N-Steps Tracing
    GIF_FREQUENCY: int or None = None
    N_OBS: int or None = None
    N_ACTS: int or None = None
    SAVE_PATH: str or None = None
    
    def __post_init__(self):
        env = gym.make(self.ENV_NAME)
        self.N_OBS, self.N_ACTS = env.observation_space.shape[0], env.action_space.shape[0]
        self.SAVE_PATH = os.path.join("saves", self.AGENT, self.EXP_NAME)


def unpack_batch(
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
