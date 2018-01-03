import numpy as np

import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from pysc2.lib import actions
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib.actions import TYPES
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType
from pysc2.env.sc2_env import SC2Env as PySC2Env


class SC2Env(gym.Env):

    def __init__(self,
                 map_name,
                 step_mul=8,
                 game_steps_per_episode=0,
                 screen_size_px=(64, 64),
                 select_army_freq=5):
        self._select_army_freq = select_army_freq
        self._screen_size_px = screen_size_px
        self._num_steps = 0
        self._sc2_env = PySC2Env(
            map_name=map_name,
            step_mul=step_mul,
            game_steps_per_episode=game_steps_per_episode,
            screen_size_px=screen_size_px,
            minimap_size_px=screen_size_px,
            visualize=False)
        self.action_spec = self._get_action_spec()

    def _step(self, action):
        function_id = action[0]
        function_args = []
        for arg_val, arg_info in zip(action[1:], FUNCTIONS[function_id].args):
            if len(arg_info.sizes) == 2:
                coords = np.unravel_index(arg_val, self._screen_size_px)
                function_args.append(coords[::-1])
            elif len(arg_info.sizes) == 1:
                function_args.append([arg_val])
        print(function_id, function_args)
        op = actions.FunctionCall(function_id, function_args)
        timestep = self._sc2_env.step([op])[0]
        #if function_id == 1:
            #assert False
        return self._transform_observation(timestep)

    def _reset(self):
        timestep = self._sc2_env.reset()[0]
        info = timestep.observation["available_actions"]
        return self._transform_observation(timestep)[0], info

    def _close(self):
        self._sc2_env.close()

    def _transform_observation(self, timestep):
        obs_screen = self._transform_spatial_features(
            timestep.observation["screen"], SCREEN_FEATURES)
        obs_minimap = self._transform_spatial_features(
            timestep.observation["minimap"], MINIMAP_FEATURES)
        obs = (obs_screen, obs_minimap)
        done = timestep.last()
        info = timestep.observation["available_actions"]
        return obs, timestep.reward, done, info

    def _transform_spatial_features(self, obs, specs):
        features = []
        for ob, spec in zip(obs, specs):
            if spec.type == FeatureType.CATEGORICAL:
                features.append(
                    np.eye(spec.scale, dtype=np.float32)[ob][:, :, 1:])
            else:
                features.append(
                    np.expand_dims(np.log(ob + 1, dtype=np.float32), axis=2))
        return np.transpose(np.concatenate(features, axis=2), (2, 0, 1))

    def _get_action_spec(self):
        action_head_sizes = []
        for argument in TYPES:
            if len(argument.sizes) == 2:
                action_head_sizes.append(
                    self._screen_size_px[0] * self._screen_size_px[1])
            elif len(argument.sizes) == 1:
                action_head_sizes.append(argument.sizes[0])
            else:
                raise NotImplementedError
        action_args_map = []
        for func in FUNCTIONS:
            action_args_map.append([arg.id for arg in func.args])
        return len(FUNCTIONS), action_head_sizes, action_args_map
