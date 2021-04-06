import dataclasses

import gym
import rc_gym
import torch
import os

import wandb


@dataclasses.dataclass
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
    GIF_FREQUENCY: int= None
    MAX_EPISODE_STEPS: int = None
    N_OBS: int= None
    N_ACTS: int= None
    SAVE_PATH: str = None
    DEVICE: str = None
    TOTAL_GRAD_STEPS: int = None

    def to_dict(self):
        return self.__dict__
    
    def __post_init__(self):
        env = gym.make(self.ENV_NAME)
        self.N_OBS, self.N_ACTS, self.MAX_EPISODE_STEPS = env.observation_space.shape[0], env.action_space.shape[0], env.spec.max_episode_steps
        self.SAVE_PATH = os.path.join("saves", self.ENV_NAME, self.AGENT, self.EXP_NAME)
        self.CHECKPOINT_PATH = os.path.join(self.SAVE_PATH, "checkpoints")
        self.GIF_PATH = os.path.join(self.SAVE_PATH, "gifs")
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(self.GIF_PATH, exist_ok=True)
        


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
    states_v = torch.Tensor(states).to(device, non_blocking=True)
    actions_v = torch.Tensor(actions).to(device, non_blocking=True)
    rewards_v = torch.Tensor(rewards).to(device, non_blocking=True)
    last_states_v = torch.Tensor(last_states).to(device, non_blocking=True)
    dones_t = torch.BoolTensor(dones).to(device, non_blocking=True)
    return states_v, actions_v, rewards_v, dones_t, last_states_v

def save_checkpoint(
    hp,
    metrics,
    pi,
    Q,
    pi_opt,
    Q_opt
):
    checkpoint = dataclasses.asdict(hp)
    checkpoint.update(metrics)
    checkpoint.update({
        "pi_state_dict": pi.state_dict(),
        "Q_state_dict": Q.state_dict(),
        "pi_opt_state_dict": pi_opt.state_dict(),
        "Q_opt_state_dict": Q_opt.state_dict(),
    })
    filename = os.path.join(
        hp.CHECKPOINT_PATH, "checkpoint_{:09}.pth".format(metrics['n_grads']))
    torch.save(checkpoint, filename)
