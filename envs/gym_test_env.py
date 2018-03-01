import numpy as np

import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

import random
import scipy.ndimage as ndimage

from pysc2.lib import actions
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib.actions import TYPES
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType
from pysc2.env.sc2_env import SC2Env as PySC2Env
from PIL import Image
import torchvision.transforms as T
import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class CarPoleEnv(gym.Env):

    def __init__(self):
        self._env = gym.make('CartPole-v0').unwrapped

    def _step(self, action):
        _, reward, done, _ = self._env.step(action[0])
        current_screen = self._get_screen()
        observation = current_screen - self._last_obs
        self._last_screen = current_screen
        return (observation, reward, done, {})

    def _get_cart_location(self):
        world_width = self._env.x_threshold * 2
        SCREEN_WIDTH = 600
        scale = SCREEN_WIDTH / world_width
        return int(self._env.state[0] * scale + SCREEN_WIDTH / 2.0)  # MIDDLE OF CART

    def _get_screen(self):
        screen = self._env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = screen[:, 160:320]
        VIEW_WIDTH = 320
        SCREEN_WIDTH = 600
        cart_location = self._get_cart_location()
        if cart_location < VIEW_WIDTH // 2:
            slice_range = slice(VIEW_WIDTH)
        elif cart_location > (SCREEN_WIDTH - VIEW_WIDTH // 2):
            slice_range = slice(-VIEW_WIDTH, None)
        else:
            slice_range = slice(cart_location - VIEW_WIDTH // 2,
                                cart_location + VIEW_WIDTH // 2)
        screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        resized_screen = resize(screen).type(Tensor)
        return resized_screen.numpy()

    def _reset(self):
        self._env.reset()
        self._last_obs = self._get_screen()
        self._num_steps = 0
        return ((self._last_obs * 0.0, self._last_obs * 0.0, self._last_obs * 0.0), None)

    def _close(self):
        self._env.close()
