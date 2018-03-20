import gym


class RewardClipWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super(RewardClipWrapper, self).__init__(env)

    def _reward(self, reward):
        if reward > 1.0:
            return 1.0
        elif reward < -1.0:
            return -1.0
        else:
            return reward
