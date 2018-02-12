from collections import namedtuple
import random

Transition = namedtuple(
    'Transition',
    ('observation', 'action', 'reward', 'next_observation', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def push(self, *args):
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)
