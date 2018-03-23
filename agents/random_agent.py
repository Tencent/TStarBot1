from gym import spaces
from envs.space import MaskableDiscrete


class RandomAgent(object):
    '''Random agent.'''

    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, observation, availables=None, eps=0):
        if availables is None:
            return self._action_space.sample()
        else:
            assert isinstance(self._action_space, MaskableDiscrete)
            return self._action_space.sample(availables)
