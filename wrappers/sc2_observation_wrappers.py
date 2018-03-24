import numpy as np

from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType

import gym
from gym import spaces

from envs.space import PySC2RawObservation


class SC2ObservationWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 unit_type_whitelist=None,
                 observation_filter=[],
                 flip=True):
        super(SC2ObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)
        self._unit_type_map = None
        if unit_type_whitelist is not None:
            self._unit_type_map = {
                v : i for i, v in enumerate(unit_type_whitelist)}
        self._observation_filter = set(observation_filter)
        self._flip = flip

        n_channels_screen = self._get_num_spatial_channels(SCREEN_FEATURES)
        n_channels_minimap = self._get_num_spatial_channels(MINIMAP_FEATURES)
        shape_screen = self.env.observation_space.space_attr["screen"][1:]
        shape_minimap = self.env.observation_space.space_attr["minimap"][1:]
        self.observation_space = spaces.Tuple([
            spaces.Box(0.0, float('inf'), [n_channels_screen, *shape_screen]),
            spaces.Box(0.0, float('inf'), [n_channels_minimap, *shape_minimap]),
            spaces.Box(0.0, float('inf'), [10])])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._observation(observation), reward, done, info

    def reset(self):
        observation, info = self.env.reset()
        return self._observation(observation), info

    def _observation(self, observation):
        observation_screen = self._transform_spatial_features(
            observation["screen"], SCREEN_FEATURES)
        observation_minimap = self._transform_spatial_features(
            observation["minimap"], MINIMAP_FEATURES)
        observation_player = self._transform_player_features(
            observation["player"])
        if self._flip:
            observation_screen = self._diagonal_flip(observation_screen)
            observation_minimap = self._diagonal_flip(observation_minimap)
        return (observation_screen, observation_minimap, observation_player)

    def _diagonal_flip(self, observation):
        if self.env.player_corner == 0:
            return np.flip(np.flip(observation, axis=1), axis=2).copy()
        else:
            return observation

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
                if spec.name == "unit_hit_points":
                    scale = 10000 # not 1600, seems a bug in pysc2
                features.append(
                    np.expand_dims(ob.astype(np.float32) / scale, axis=2))
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

    @property
    def player_corner(self):
        return self.env.player_corner


class SC2ObservationNonSpatialWrapperV0(gym.ObservationWrapper):

    def __init__(self, env):
        super(SC2ObservationNonSpatialWrapperV0, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)
        self.observation_space = spaces.Box(0.0, float('inf'), [10])

    def _observation(self, observation):
        observation_player = self._transform_player_features(
            observation["player"])
        return observation_player

    def _transform_player_features(self, observation):
        return np.log10(observation[1:].astype(np.float32) + 1)

    @property
    def player_corner(self):
        return self.env.player_corner


PLAYER_RELATIVE_ENEMY = 4
PLAYER_RELATIVE_SELF = 1


class SC2ObservationNonSpatialWrapperV1(gym.Wrapper):

    def __init__(self, env):
        super(SC2ObservationNonSpatialWrapperV1, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)
        self.observation_space = spaces.Box(0.0, float('inf'), [22])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._observation(observation), reward, done, info

    def reset(self):
        observation, info = self.env.reset()
        self._update_player_position_mask(observation)
        return self._observation(observation), info

    def _observation(self, observation):
        observation_player = observation["player"]
        player_id = observation_player[0]
        minerals = observation_player[1]
        vespene = observation_player[2]
        food_used = observation_player[3]
        food_cap = observation_player[4]
        food_army = observation_player[5]
        food_workers = observation_player[6]
        idle_worker_count = observation_player[7]
        army_count = observation_player[8]
        warp_gate_count = observation_player[9]
        larva_count = observation_player[10]

        game_loop = observation["game_loop"][0]
        enemy_pixels_minimap = self._count_enemy_pixels_minimap(observation)
        enemy_pixels_screen = self._count_enemy_pixels_screen(observation)
        enemy_pixels_self_range = self._count_enemy_pixels_self_range(observation)
        enemy_pixels_oppo_range = self._count_enemy_pixels_oppo_range(observation)
        self_pixels_minimap = self._count_self_pixels_minimap(observation)
        self_pixels_screen = self._count_self_pixels_screen(observation)
        self_pixels_self_range = self._count_self_pixels_self_range(observation)
        self_pixels_oppo_range = self._count_self_pixels_oppo_range(observation)

        features = np.array([minerals,
                             vespene,
                             food_used,
                             food_cap,
                             food_army,
                             food_workers,
                             idle_worker_count,
                             army_count,
                             larva_count,
                             game_loop,
                             enemy_pixels_minimap,
                             enemy_pixels_screen,
                             enemy_pixels_self_range,
                             enemy_pixels_oppo_range,
                             self_pixels_minimap,
                             self_pixels_screen,
                             self_pixels_self_range,
                             self_pixels_oppo_range,
                             self.env.num_spawning_pools,
                             self.env.num_extractors,
                             self.env.num_roach_warrens,
                             self.env.num_queens]).astype(np.float32)
        log_features = np.log10(features + 1)
        assert log_features.shape[0] == self.observation_space.shape[0]
        return log_features

    def _count_enemy_pixels_minimap(self, observation):
        player_relative = observation["minimap"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(player_relative == PLAYER_RELATIVE_ENEMY)

    def _count_enemy_pixels_screen(self, observation):
        player_relative = observation["screen"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(player_relative == PLAYER_RELATIVE_ENEMY)

    def _count_enemy_pixels_self_range(self, observation):
        player_relative = observation["minimap"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(np.logical_and(player_relative == PLAYER_RELATIVE_ENEMY,
                                     self._self_range_mask == 1))

    def _count_enemy_pixels_oppo_range(self, observation):
        player_relative = observation["minimap"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(np.logical_and(player_relative == PLAYER_RELATIVE_ENEMY,
                                     self._opponent_range_mask == 1))

    def _count_self_pixels_minimap(self, observation):
        player_relative = observation["minimap"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(player_relative == PLAYER_RELATIVE_SELF)

    def _count_self_pixels_screen(self, observation):
        player_relative = observation["screen"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(player_relative == PLAYER_RELATIVE_SELF)

    def _count_self_pixels_self_range(self, observation):
        player_relative = observation["minimap"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(np.logical_and(player_relative == PLAYER_RELATIVE_SELF,
                                     self._self_range_mask == 1))

    def _count_self_pixels_oppo_range(self, observation):
        player_relative = observation["minimap"][
            MINIMAP_FEATURES.player_relative.index]
        return np.sum(np.logical_and(player_relative == PLAYER_RELATIVE_SELF,
                                     self._opponent_range_mask == 1))

    def _update_player_position_mask(self, observation):
        camera = observation["minimap"][MINIMAP_FEATURES.camera.index]
        resolution = self.env.observation_space.space_attr["minimap"][-1]
        half_resolution = resolution // 2
        if np.nonzero(camera == 1)[0].any() < resolution / 2:
            self._self_range_mask = np.zeros((resolution, resolution))
            self._opponent_range_mask = np.zeros((resolution, resolution))
            self._self_range_mask[:half_resolution, :half_resolution] = 1
            self._opponent_range_mask[-half_resolution:, -half_resolution:] = 1
        else:
            print("position right-bottom.")
            self._self_range_mask = np.zeros((resolution, resolution))
            self._opponent_range_mask = np.zeros((resolution, resolution))
            self._self_range_mask[-half_resolution:, -half_resolution:] = 1
            self._oppenent_range_mask[:half_resolution, :half_resolution] = 1

    @property
    def player_corner(self):
        return self.env.player_corner
