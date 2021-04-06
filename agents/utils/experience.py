from collections import namedtuple, deque
from itertools import islice
import numpy as np

ExperienceFirstLast = namedtuple(
    'ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])

# based on coax and ptan
class NStepTracer():
    """
    A short-term cache for n-step bootstrapping.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.
    """

    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._deque_s = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = np.power(self.gamma, np.arange(self.n))
        self._gamman = np.power(self.gamma, self.n)

    def add(self, s, a, r, done):
        if self._done and len(self):
            # ("please flush cache (or repeatedly call popleft) before appending new transitions")
            raise Exception

        self._deque_s.append((s, a))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_s)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        if not self:
            # ("cache needs to receive more transitions before it can be popped from")
            raise Exception

        # pop state-action (propensities) pair
        s, a = self._deque_s.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        self._deque_r.popleft()

        # keep in mind that we've already popped (s, a)
        if len(self) >= self.n:
            s_next, a_next = self._deque_s[self.n - 1]
            done = False
        else:
            # no more bootstrapping
            s_next, a_next, done = None, a, True

        return ExperienceFirstLast(state=s,
                                   action=a,
                                   reward=rn,
                                   last_state=s_next)


class ExperienceReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def add(self, sample):
        assert(isinstance(sample, ExperienceFirstLast))
        self.buffer.append(sample)
