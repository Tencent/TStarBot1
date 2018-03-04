class RandomAgent(object):
    '''Random agent.'''

    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, observation, eps=0):
        return self._action_space.sample()
