import numpy as np

from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType

import gym
from gym import spaces

from envs.space import PySC2ObservationSpace


class SC2ObservationWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env,
                 unit_type_whitelist=None,
                 observation_filter=[]):
        super(SC2ObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2ObservationSpace)
        self._unit_type_map = None
        if unit_type_whitelist is not None:
            self._unit_type_map = {
                v : i for i, v in enumerate(unit_type_whitelist)}
        self._observation_filter = set(observation_filter)

        n_channels_screen = self._get_num_spatial_channels(SCREEN_FEATURES)
        n_channels_minimap = self._get_num_spatial_channels(MINIMAP_FEATURES)
        shape_screen = self.env.observation_space.space_attr["screen"][1:]
        shape_minimap = self.env.observation_space.space_attr["minimap"][1:]
        self.observation_space = spaces.Tuple([
            spaces.Box(0.0, float('inf'), [n_channels_screen, *shape_screen]),
            spaces.Box(0.0, float('inf'), [n_channels_minimap, *shape_minimap]),
            spaces.Box(0.0, float('inf'), [10])])

    def _observation(self, observation):
        observation_screen = self._transform_spatial_features(
            observation["screen"], SCREEN_FEATURES)
        observation_minimap = self._transform_spatial_features(
            observation["minimap"], MINIMAP_FEATURES)
        observation_player = self._transform_player_features(
            observation["player"])
        return (observation_screen, observation_minimap, observation_player)

    def _transform_spatial_features(self, observation, specs):
        features = []
        for ob, spec in zip(observation, specs):
            if spec.name in self._observation_filter:
                continue
            scale = spec.scale
            if spec.name == "unit_type" and self._unit_type_map is not None:
                ob = np.vectorize(lambda k: self._unit_type_map.get(k, 0))(ob)
                scale = len(self._unit_type_map)
            if spec.type == FeatureType.CATEGORICAL:
                features.append(np.eye(scale, dtype=np.float32)[ob][:, :, 1:])
            else:
                features.append(
                    np.expand_dims(np.log10(ob + 1, dtype=np.float32), axis=2))
        return np.transpose(np.concatenate(features, axis=2), (2, 0, 1))

    def _transform_player_features(self, observation):
        return np.log10(observation[1:].astype(np.float32) + 1)

    def _get_num_spatial_channels(self, specs):
        num_channels = 0
        for spec in specs:
            if spec.name in self._observation_filter:
                continue
            scale = spec.scale
            if spec.name == "unit_type" and self._unit_type_map is not None:
                scale =  len(self._unit_type_map)
            if spec.type == FeatureType.CATEGORICAL:
                num_channels += scale - 1
            else:
                num_channels += 1
        return num_channels


class SC2ObservationTinyWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(SC2ObservationTinyWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2ObservationSpace)
        self.observation_space = spaces.Box(0.0, float('inf'), [10])

    def _observation(self, observation):
        observation_player = self._transform_player_features(
            observation["player"])
        return observation_player

    def _transform_player_features(self, observation):
        return np.log10(observation[1:].astype(np.float32) + 1)
