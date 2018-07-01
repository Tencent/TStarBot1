from gym import spaces
from sc2learner.envs.space import MaskDiscrete
from sc2learner.envs.space import PySC2RawAction
import numpy as np


class RandomAgent(object):
    '''Random agent.'''

    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, observation, eps=0):
        if (isinstance(self._action_space, MaskDiscrete) or
            isinstance(self._action_space, PySC2RawAction)):
            action_mask = observation[-1]
            return self._action_space.sample(np.nonzero(action_mask)[0])
        else:
            return self._action_space.sample()
