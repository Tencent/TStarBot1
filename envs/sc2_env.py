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
                 select_army_freq=5,
                 action_filter=[],
                 observation_filter=[]):
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
        self._valid_action_ids = list(set(range(524)) - set(action_filter))
        self._observation_filter = set(observation_filter) 
    
    @property
    def action_spec(self):
        return self._get_action_spec()

    @property
    def observation_spec(self):
        return self._get_observation_spec()

    def _step(self, action):
        function_id = self._valid_action_ids[action[0]]
        function_args = []
        for arg_val, arg_info in zip(action[1:], FUNCTIONS[function_id].args):
            if len(arg_info.sizes) == 2:
                coords = np.unravel_index(arg_val, self._screen_size_px)
                function_args.append(coords[::-1])
            elif len(arg_info.sizes) == 1:
                function_args.append([arg_val])
        op = actions.FunctionCall(function_id, function_args)
        timestep = self._sc2_env.step([op])[0]
        return self._transform_observation(timestep)

    def _reset(self):
        timestep = self._sc2_env.reset()[0]
        return (self._transform_observation(timestep)[0],
                self._transform_observation(timestep)[3])

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
        info = [self._valid_action_ids.index(fid)
                for fid in info if fid in self._valid_action_ids]
        return obs, timestep.reward, done, info

    def _transform_spatial_features(self, obs, specs):
        features = []
        for ob, spec in zip(obs, specs):
            if spec.name in self._observation_filter:
                continue
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
        for func_id in self._valid_action_ids:
            action_args_map.append([arg.id for arg in FUNCTIONS[func_id].args])
        return len(self._valid_action_ids), action_head_sizes, action_args_map

    def _get_observation_spec(self):
        def get_spatial_channels(specs):
            num_channels = 0
            for spec in specs:
                if spec.name in self._observation_filter:
                    continue
                if spec.type == FeatureType.CATEGORICAL:
                    num_channels += spec.scale - 1
                else:
                    num_channels += 1
            return num_channels
        num_channels_screen = get_spatial_channels(SCREEN_FEATURES)
        num_channels_minimap = get_spatial_channels(MINIMAP_FEATURES)
        return num_channels_screen, num_channels_minimap
