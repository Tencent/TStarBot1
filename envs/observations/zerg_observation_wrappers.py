import numpy as np
import gym
from gym import spaces

from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

from envs.space import PySC2RawObservation
from envs.space import MaskableDiscrete
from envs.observations.spatial_features import UnitTypeCountMapFeature
from envs.observations.spatial_features import AllianceCountMapFeature
from envs.observations.nonspatial_features import PlayerFeature
from envs.observations.nonspatial_features import UnitTypeCountFeature
from envs.observations.nonspatial_features import UnitStatCountFeature
from envs.observations.nonspatial_features import GameProgressFeature
from envs.observations.nonspatial_features import ActionSeqFeature


class ZergObservationWrapper(gym.Wrapper):

    def __init__(self, env, flip=True):
        super(ZergObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)
        resolution = self.env.observation_space.space_attr["minimap"][1]

        # spatial features
        self._unit_type_count_map_feature = UnitTypeCountMapFeature(
            type_map={UNIT_TYPE.ZERG_DRONE.value: 0,
                      UNIT_TYPE.ZERG_ZERGLING.value: 1,
                      UNIT_TYPE.ZERG_ROACH.value: 2,
                      UNIT_TYPE.ZERG_ROACHBURROWED.value: 2,
                      UNIT_TYPE.ZERG_HYDRALISK.value: 3,
                      UNIT_TYPE.ZERG_OVERLORD.value: 4,
                      UNIT_TYPE.ZERG_OVERSEER.value: 4,
                      UNIT_TYPE.ZERG_HATCHERY.value: 5,
                      UNIT_TYPE.ZERG_LAIR.value: 5,
                      UNIT_TYPE.ZERG_HIVE.value: 5,
                      UNIT_TYPE.ZERG_EXTRACTOR.value: 6,
                      UNIT_TYPE.ZERG_QUEEN.value: 7,
                      UNIT_TYPE.ZERG_RAVAGER.value: 8,
                      UNIT_TYPE.ZERG_BANELING.value: 9,
                      UNIT_TYPE.ZERG_LURKERMP.value: 10,
                      UNIT_TYPE.ZERG_LURKERMPBURROWED.value: 10,
                      UNIT_TYPE.ZERG_VIPER.value: 11,
                      UNIT_TYPE.ZERG_MUTALISK.value: 12,
                      UNIT_TYPE.ZERG_CORRUPTOR.value: 13,
                      UNIT_TYPE.ZERG_BROODLORD.value: 14,
                      UNIT_TYPE.ZERG_SWARMHOSTMP.value: 15,
                      UNIT_TYPE.ZERG_INFESTOR.value: 16,
                      UNIT_TYPE.ZERG_ULTRALISK.value: 17,
                      UNIT_TYPE.ZERG_CHANGELING.value: 18,
                      UNIT_TYPE.ZERG_SPINECRAWLER.value: 19,
                      UNIT_TYPE.ZERG_SPORECRAWLER.value: 20},
            resolution=resolution)
        self._alliance_count_map_feature = AllianceCountMapFeature(resolution)

        # nonspatial features
        self._unit_count_feature = UnitTypeCountFeature(
            type_list=[UNIT_TYPE.ZERG_LARVA.value,
                       UNIT_TYPE.ZERG_DRONE.value,
                       UNIT_TYPE.ZERG_ZERGLING.value,
                       UNIT_TYPE.ZERG_BANELING.value,
                       UNIT_TYPE.ZERG_ROACH.value,
                       UNIT_TYPE.ZERG_ROACHBURROWED.value,
                       UNIT_TYPE.ZERG_RAVAGER.value,
                       UNIT_TYPE.ZERG_HYDRALISK.value,
                       UNIT_TYPE.ZERG_LURKERMP.value,
                       UNIT_TYPE.ZERG_LURKERMPBURROWED.value,
                       UNIT_TYPE.ZERG_VIPER.value,
                       UNIT_TYPE.ZERG_MUTALISK.value,
                       UNIT_TYPE.ZERG_CORRUPTOR.value,
                       UNIT_TYPE.ZERG_BROODLORD.value,
                       UNIT_TYPE.ZERG_SWARMHOSTMP.value,
                       UNIT_TYPE.ZERG_LOCUSTMP.value,
                       UNIT_TYPE.ZERG_INFESTOR.value,
                       UNIT_TYPE.ZERG_ULTRALISK.value,
                       UNIT_TYPE.ZERG_BROODLING.value,
                       UNIT_TYPE.ZERG_OVERLORD.value,
                       UNIT_TYPE.ZERG_OVERSEER.value,
                       UNIT_TYPE.ZERG_QUEEN.value,
                       UNIT_TYPE.ZERG_CHANGELING.value,
                       UNIT_TYPE.ZERG_SPINECRAWLER.value,
                       UNIT_TYPE.ZERG_SPORECRAWLER.value,
                       UNIT_TYPE.ZERG_NYDUSCANAL.value,
                       UNIT_TYPE.ZERG_EXTRACTOR.value,
                       UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                       UNIT_TYPE.ZERG_ROACHWARREN.value,
                       UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                       UNIT_TYPE.ZERG_HATCHERY.value,
                       UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                       UNIT_TYPE.ZERG_BANELINGNEST.value,
                       UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                       UNIT_TYPE.ZERG_SPIRE.value,
                       UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                       UNIT_TYPE.ZERG_NYDUSNETWORK.value,
                       UNIT_TYPE.ZERG_LURKERDENMP.value,
                       UNIT_TYPE.ZERG_LAIR.value,
                       UNIT_TYPE.ZERG_HIVE.value,
                       UNIT_TYPE.ZERG_GREATERSPIRE.value])
        self._unit_stat_count_feature = UnitStatCountFeature()
        self._player_feature = PlayerFeature()
        self._game_progress_feature = GameProgressFeature()
        self._action_seq_feature = ActionSeqFeature(self.action_space.n, 10)

        n_channels = sum([self._unit_type_count_map_feature.num_channels,
                          self._alliance_count_map_feature.num_channels])
        n_dims = sum([self._unit_stat_count_feature.num_dims,
                      self._unit_count_feature.num_dims,
                      self._player_feature.num_dims,
                      self._game_progress_feature.num_dims,
                      self._action_seq_feature.num_dims])
        self.observation_space = spaces.Tuple([
            spaces.Box(0.0, float('inf'), [n_channels, resolution, resolution]),
            spaces.Box(0.0, float('inf'), [n_dims])])
        self._flip = flip

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

        ally_map_feat = self._alliance_count_map_feature.features(observation)
        type_map_feat = self._unit_type_count_map_feature.features(observation)
        unit_type_feat = self._unit_count_feature.features(observation)
        unit_stat_feat = self._unit_stat_count_feature.features(observation)
        player_feat = self._player_feature.features(observation)
        game_progress_feat = self._game_progress_feature.features(observation)
        action_seq_feat = self._action_seq_feature.features()

        spatial_feat = np.concatenate([ally_map_feat,
                                       type_map_feat])
        nonspatial_feat = np.concatenate([unit_type_feat,
                                          unit_stat_feat,
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


class ZergNonspatialObservationWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ZergNonspatialObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)

        self._unit_count_feature = UnitTypeCountFeature(
            type_list=[UNIT_TYPE.ZERG_LARVA.value,
                       UNIT_TYPE.ZERG_DRONE.value,
                       UNIT_TYPE.ZERG_ZERGLING.value,
                       UNIT_TYPE.ZERG_BANELING.value,
                       UNIT_TYPE.ZERG_ROACH.value,
                       UNIT_TYPE.ZERG_ROACHBURROWED.value,
                       UNIT_TYPE.ZERG_RAVAGER.value,
                       UNIT_TYPE.ZERG_HYDRALISK.value,
                       UNIT_TYPE.ZERG_LURKERMP.value,
                       UNIT_TYPE.ZERG_LURKERMPBURROWED.value,
                       UNIT_TYPE.ZERG_VIPER.value,
                       UNIT_TYPE.ZERG_MUTALISK.value,
                       UNIT_TYPE.ZERG_CORRUPTOR.value,
                       UNIT_TYPE.ZERG_BROODLORD.value,
                       UNIT_TYPE.ZERG_SWARMHOSTMP.value,
                       UNIT_TYPE.ZERG_LOCUSTMP.value,
                       UNIT_TYPE.ZERG_INFESTOR.value,
                       UNIT_TYPE.ZERG_ULTRALISK.value,
                       UNIT_TYPE.ZERG_BROODLING.value,
                       UNIT_TYPE.ZERG_OVERLORD.value,
                       UNIT_TYPE.ZERG_OVERSEER.value,
                       UNIT_TYPE.ZERG_QUEEN.value,
                       UNIT_TYPE.ZERG_CHANGELING.value,
                       UNIT_TYPE.ZERG_SPINECRAWLER.value,
                       UNIT_TYPE.ZERG_SPORECRAWLER.value,
                       UNIT_TYPE.ZERG_NYDUSCANAL.value,
                       UNIT_TYPE.ZERG_EXTRACTOR.value,
                       UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                       UNIT_TYPE.ZERG_ROACHWARREN.value,
                       UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                       UNIT_TYPE.ZERG_HATCHERY.value,
                       UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                       UNIT_TYPE.ZERG_BANELINGNEST.value,
                       UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                       UNIT_TYPE.ZERG_SPIRE.value,
                       UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                       UNIT_TYPE.ZERG_NYDUSNETWORK.value,
                       UNIT_TYPE.ZERG_LURKERDENMP.value,
                       UNIT_TYPE.ZERG_LAIR.value,
                       UNIT_TYPE.ZERG_HIVE.value,
                       UNIT_TYPE.ZERG_GREATERSPIRE.value])
        self._unit_stat_count_feature = UnitStatCountFeature()
        self._player_feature = PlayerFeature()
        self._game_progress_feature = GameProgressFeature()
        self._action_seq_feature = ActionSeqFeature(self.action_space.n, 10)

        n_dims = sum([self._unit_stat_count_feature.num_dims,
                      self._unit_count_feature.num_dims,
                      self._player_feature.num_dims,
                      self._game_progress_feature.num_dims,
                      self._action_seq_feature.num_dims])
        self.observation_space = spaces.Box(0.0, float('inf'), [n_dims])

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

        unit_type_feat = self._unit_count_feature.features(observation)
        unit_stat_feat = self._unit_stat_count_feature.features(observation)
        player_feat = self._player_feature.features(observation)
        game_progress_feat = self._game_progress_feature.features(observation)
        action_seq_feat = self._action_seq_feature.features()

        nonspatial_feat = np.concatenate([unit_type_feat,
                                          unit_stat_feat,
                                          player_feat,
                                          game_progress_feat,
                                          action_seq_feat])

        if isinstance(self.action_space, MaskableDiscrete):
            return (nonspatial_feat, action_mask)
        else:
            return (nonspatial_feat)
