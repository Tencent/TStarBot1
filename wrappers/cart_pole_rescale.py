from PIL import Image
import numpy as np

from gym import Wrapper
from gym.spaces.box import Box

import torch
import torchvision.transforms as T


resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


class CartPoleRescaleWrapper(Wrapper):

    def __init__(self, env):
        super(CartPoleRescaleWrapper, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [3, 40, 80])

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        current_screen = self._get_rescaled_screen()
        observation = current_screen - self._last_screen
        self._last_screen = current_screen
        return (observation, reward, done, info)

    def reset(self):
        self.env.reset()
        self._last_screen = self._get_rescaled_screen()
        np.set_printoptions(threshold=np.nan)
        return np.zeros(self._last_screen.shape, dtype=np.float32)

    def _get_rescaled_screen(self):
        VIEW_WIDTH = 320
        SCREEN_WIDTH = 600

        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = screen[:, 160:320]
        world_width = self.env.x_threshold * 2
        scale = SCREEN_WIDTH / world_width
        cart_location = int(self.env.state[0] * scale + SCREEN_WIDTH / 2.0)
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
        resized_screen = resize(screen)
        return resized_screen.numpy()
