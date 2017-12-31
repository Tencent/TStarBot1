import numpy as np

import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.env.sc2_env import SC2Env


class SC2MiniEnv(gym.Env):

    def __init__(self,
                 map_name,
                 step_mul=8,
                 game_steps_per_episode=0,
                 screen_size_px=(64, 64),
                 select_army_freq=5):
        self._sc2_env = SC2Env(
            map_name=map_name,
            step_mul=step_mul,
            game_steps_per_episode=game_steps_per_episode,
            screen_size_px=screen_size_px,
            minimap_size_px=screen_size_px,
            visualize=False)
        self.action_space = Discrete(screen_size_px[0] * screen_size_px[1])
        self.observation_space = Box(
            low=0, high=SCREEN_FEATURES.player_relative.scale,
            shape=[screen_size_px[0], screen_size_px[1], 1])
        self._select_army_freq = select_army_freq
        self._screen_size_px = screen_size_px
        self._num_steps = 0

    def _step(self, action):
        if (self._select_army_freq > 0 and
            self._num_steps % self._select_army_freq == 0):
            self._select_all_army()
        timestep = self._attack_move(action)
        self._num_steps += 1
        return self._convert_observation(timestep)

    def _reset(self):
        timestep = self._sc2_env.reset()[0]
        return self._convert_observation(timestep)

    def _close(self):
        self._sc2_env.close()

    def _select_all_army(self):
        select_army_op = actions.FunctionCall(
            actions.FUNCTIONS.select_army.id, [[0]])
        self._sc2_env.step([select_army_op])

    def _attack_move(self, action):
        coords = np.unravel_index(action, self._screen_size_px)
        action = [actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id,
                                       [[0], coords[::-1]])] # reversed coords
        try:
            timestep = self._sc2_env.step(action)[0]
        except ValueError:
            self._select_all_army()
            timestep = self._sc2_env.step(action)[0]
        return timestep

    def _convert_observation(self, timestep):
        obs = timestep.observation["screen"][
            SCREEN_FEATURES.player_relative.index]
        done = False
        #done = timestep.last()
        info = None
        return obs, timestep.reward, done, info
