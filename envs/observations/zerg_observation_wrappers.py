import numpy as np
import gym
from gym import spaces

from pysc2.lib.typeenums import UNIT_TYPEID

from envs.space import PySC2RawObservation
from envs.space import MaskableDiscrete
from envs.observations.spatial_features import UnitType3DFeature
from envs.observations.spatial_features import PlayerRelative3DFeature
from envs.observations.nonspatial_features import Player1DFeature
from envs.observations.nonspatial_features import UnitCount1DFeature
from envs.observations.nonspatial_features import UnitHasOrNotFeature
from envs.observations.nonspatial_features import UnitStat1DFeature
from envs.observations.nonspatial_features import GameProgressFeature
from envs.observations.nonspatial_features import ActionSeqFeature


class ZergObservationWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 flip=True):
        super(ZergObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)

        resolution = self.env.observation_space.space_attr["minimap"][1]
        self._unit_type_feature = UnitType3DFeature(
            type_map={UNIT_TYPEID.ZERG_DRONE.value: 0,
                      UNIT_TYPEID.ZERG_ZERGLING.value: 1,
                      UNIT_TYPEID.ZERG_ROACH.value: 2,
                      UNIT_TYPEID.ZERG_HYDRALISK.value: 3,
                      UNIT_TYPEID.ZERG_OVERLORD.value: 4,
                      UNIT_TYPEID.ZERG_OVERSEER.value: 5,
                      UNIT_TYPEID.ZERG_HATCHERY.value: 6,
                      UNIT_TYPEID.ZERG_LAIR.value: 6,
                      UNIT_TYPEID.ZERG_HIVE.value: 6,
                      UNIT_TYPEID.ZERG_EXTRACTOR.value: 7,
                      UNIT_TYPEID.ZERG_QUEEN.value: 8,
                      UNIT_TYPEID.ZERG_RAVAGER.value: 9,
                      UNIT_TYPEID.ZERG_BANELING.value: 10},
            resolution=resolution)
        self._player_relative_feature = PlayerRelative3DFeature(resolution)
        self._unit_count_feature = UnitCount1DFeature(
            type_list=[UNIT_TYPEID.ZERG_DRONE.value,
                       UNIT_TYPEID.ZERG_ZERGLING.value,
                       UNIT_TYPEID.ZERG_ROACH.value,
                       UNIT_TYPEID.ZERG_HYDRALISK.value,
                       UNIT_TYPEID.ZERG_OVERLORD.value,
                       UNIT_TYPEID.ZERG_OVERSEER.value,
                       UNIT_TYPEID.ZERG_QUEEN.value,
                       UNIT_TYPEID.ZERG_CHANGELINGZERGLING.value,
                       UNIT_TYPEID.ZERG_RAVAGER.value,
                       UNIT_TYPEID.ZERG_EGG.value,
                       UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value,
                       UNIT_TYPEID.ZERG_BANELING.value,
                       UNIT_TYPEID.ZERG_BROODLING.value,
                       UNIT_TYPEID.ZERG_LARVA.value,
                       UNIT_TYPEID.ZERG_HATCHERY.value,
                       UNIT_TYPEID.ZERG_LAIR.value,
                       UNIT_TYPEID.ZERG_HIVE.value,
                       UNIT_TYPEID.ZERG_BANELINGNEST.value,
                       UNIT_TYPEID.ZERG_SPAWNINGPOOL.value,
                       UNIT_TYPEID.ZERG_ROACHWARREN.value,
                       UNIT_TYPEID.ZERG_HYDRALISKDEN.value,
                       UNIT_TYPEID.ZERG_EXTRACTOR.value])
        self._unit_has_feature = UnitHasOrNotFeature(
            type_list=[UNIT_TYPEID.ZERG_LARVA.value])
        self._unit_stat_feature = UnitStat1DFeature()
        self._player_feature = Player1DFeature()
        self._game_progress_feature = GameProgressFeature()
        self._action_seq_feature = ActionSeqFeature(self.action_space.n, 15)
        self._flip = flip

        n_channels = sum([self._unit_type_feature.num_channels,
                          self._player_relative_feature.num_channels])
        n_dims = sum([self._unit_stat_feature.num_dims,
                      self._unit_count_feature.num_dims,
                      self._unit_has_feature.num_dims,
                      self._player_feature.num_dims,
                      self._game_progress_feature.num_dims,
                      self._action_seq_feature.num_dims])
        self.observation_space = spaces.Tuple([
            spaces.Box(0.0, float('inf'), [n_channels, resolution, resolution]),
            spaces.Box(0.0, float('inf'), [n_dims])])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._action_seq_feature.push_action(action)
        return self._observation(observation), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._action_seq_feature.reset()
        return self._observation(observation)

    @property
    def action_names(self):
        if not hasattr(self.env, 'action_names'):
            raise NotImplementedError
        return self.env.action_names

    @property
    def player_position(self):
        if not hasattr(self.env, 'player_position'):
            raise NotImplementedError
        return self.env.player_position

    def _observation(self, observation):
        if isinstance(self.env.action_space, MaskableDiscrete):
            observation, action_mask = observation

        player_rel_feat = self._player_relative_feature.features(observation)
        unit_type_feat = self._unit_type_feature.features(observation)
        unit_count_feat = self._unit_count_feature.features(observation)
        unit_has_feat = self._unit_has_feature.features(observation)
        unit_stat_feat = self._unit_stat_feature.features(observation)
        player_feat = self._player_feature.features(observation)
        game_progress_feat = self._game_progress_feature.features(observation)
        action_seq_feat = self._action_seq_feature.features()

        spatial_feat = np.concatenate([player_rel_feat,
                                       unit_type_feat])
        nonspatial_feat = np.concatenate([unit_stat_feat,
                                          unit_count_feat,
                                          unit_has_feat,
                                          player_feat,
                                          game_progress_feat,
                                          action_seq_feat])

        #np.set_printoptions(threshold=np.nan, linewidth=300)
        #for i in range(spatial_feat.shape[0]):
            #print(spatial_feat[i])

        if self._flip:
            spatial_feat = self._diagonal_flip(spatial_feat)

        if isinstance(self.action_space, MaskableDiscrete):
            return (spatial_feat, nonspatial_feat, action_mask)
        else:
            return (spatial_feat, nonspatial_feat)

    def _diagonal_flip(self, feature):
        if self.env.player_position == 0:
            return np.flip(np.flip(feature, axis=1), axis=2).copy()
        else:
            return feature
