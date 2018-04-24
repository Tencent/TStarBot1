import numpy as np
from enum import Enum, unique

from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType

import gym
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID

from envs.space import PySC2RawObservation
from envs.space import MaskableDiscrete

@unique
class AllianceType(Enum):
    SELF = 1
    ALLY = 2
    NEUTRAL = 3
    ENEMY = 4


class UnitType3DFeature(object):

    def __init__(self, type_map, resolution, world_size=(200.0, 176.0)):
        self._type_map = type_map
        self._resolution = resolution
        self._world_size = world_size

    def features(self, observation):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == AllianceType.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == AllianceType.ENEMY.value]
        self_features = self._generate_features(self_units)
        enemy_features = self._generate_features(enemy_units)
        return np.concatenate((self_features, enemy_features))

    @property
    def num_channels(self):
        return (max(self._type_map.values()) + 1) * 2

    def _generate_features(self, units):
        num_channels = max(self._type_map.values()) + 1
        features = np.zeros((num_channels, self._resolution, self._resolution),
                            dtype=np.float32)
        grid_width = self._world_size[0] / self._resolution
        grid_height = self._world_size[1] / self._resolution
        for u in units:
            if u.unit_type in self._type_map:
                c = self._type_map[u.unit_type]
                x = u.float_attr.pos_x // grid_width
                y = self._resolution - 1 - u.float_attr.pos_y // grid_height
                features[c, int(y), int(x)] += 1.0
        return features


class PlayerRelative3DFeature(object):

    def __init__(self, resolution, world_size=(200.0, 176.0)):
        self._resolution = resolution
        self._world_size = world_size

    def features(self, observation):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == AllianceType.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == AllianceType.ENEMY.value]
        neutral_units = [u for u in observation['units']
                         if u.int_attr.alliance == AllianceType.NEUTRAL.value]
        self_features = self._generate_features(self_units)
        enemy_features = self._generate_features(enemy_units)
        neutral_features = self._generate_features(neutral_units)
        return np.concatenate((self_features, enemy_features, neutral_features))

    @property
    def num_channels(self):
        return 3

    def _generate_features(self, units):
        features = np.zeros((1, self._resolution, self._resolution),
                             dtype=np.float32)
        grid_width = self._world_size[0] / self._resolution
        grid_height = self._world_size[1] / self._resolution
        for u in units:
            x = u.float_attr.pos_x // grid_width
            y = self._resolution - 1 - u.float_attr.pos_y // grid_height
            features[0, int(y), int(x)] += 1.0
        return features


class Player1DFeature(object):

    def features(self, observation):
        player_features = observation["player"][1:-1].astype(np.float32)
        food_unused = player_features[3] - player_features[2]
        player_features[-1] = food_unused if food_unused >= 0 else 0
        scale = np.array([1000, 1000, 10, 10, 10, 10, 10, 10, 10])
        scaled_features = (player_features / scale).astype(np.float32)
        log_features = np.log10(player_features + 1).astype(np.float32)

        additional_features = np.zeros(10, dtype=np.float32)
        if food_unused <= 0:
            additional_features[0] = 1
        elif food_unused <= 3:
            additional_features[1] = 1
        elif food_unused <= 6:
            additional_features[2] = 1
        elif food_unused <= 9:
            additional_features[3] = 1
        elif food_unused <= 12:
            additional_features[4] = 1
        elif food_unused <= 15:
            additional_features[5] = 1
        elif food_unused <= 18:
            additional_features[6] = 1
        elif food_unused <= 21:
            additional_features[7] = 1
        elif food_unused <= 24:
            additional_features[8] = 1
        else:
            additional_features[9] = 1
        return np.concatenate((scaled_features, log_features,
                               additional_features))

    @property
    def num_dims(self):
        return 9 * 2 + 10


class UnitCount1DFeature(object):

    def __init__(self, type_list):
        self._type_list = type_list

    def features(self, observation):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == AllianceType.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == AllianceType.ENEMY.value]
        self_features = self._generate_features(self_units)
        enemy_features = self._generate_features(enemy_units)
        features = np.concatenate((self_features, enemy_features))
        scaled_features = features / 10
        log_features = np.log10(features + 1)
        return np.concatenate((scaled_features, log_features))


    @property
    def num_dims(self):
        return len(self._type_list) * 2 * 2

    def _generate_features(self, units):
        count = {t: 0 for t in self._type_list}
        for u in units:
            if u.unit_type in count:
                count[u.unit_type] += 1
        return np.array(list(count.values()), dtype=np.float32)


class UnitHasOrNotFeature(object):

    def __init__(self, type_list):
        self._type_list = type_list

    def features(self, observation):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == AllianceType.SELF.value]
        features = self._generate_features(self_units)
        return features


    @property
    def num_dims(self):
        return len(self._type_list)

    def _generate_features(self, units):
        count = {t: 0 for t in self._type_list}
        for u in units:
            if u.unit_type in count:
                count[u.unit_type] = 1
        return np.array(list(count.values()), dtype=np.float32)


class UnitStat1DFeature(object):

    def features(self, observation):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == AllianceType.SELF.value]
        self_flying_units = [u for u in self_units if u.bool_attr.is_flying]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == AllianceType.ENEMY.value]
        enemy_flying_units = [u for u in enemy_units if u.bool_attr.is_flying]

        features = np.array([len(self_units),
                             len(self_flying_units),
                             len(enemy_units),
                             len(enemy_flying_units)], dtype=np.float32)
        scaled_features = features / 50
        log_features = np.log10(features + 1)
        return np.concatenate((scaled_features, log_features))

    @property
    def num_dims(self):
        return 4 * 2


class GameProgressFeature(object):

    def features(self, observation):
        game_loop = observation["game_loop"][0]
        features_20 = self._onehot(game_loop, 20)
        features_8 = self._onehot(game_loop, 8)
        features_5 = self._onehot(game_loop, 5)
        return np.concatenate([features_20, features_8, features_5])

    def _onehot(self, value, n_bins):
        bin_width = 24000 // n_bins
        features = np.zeros(n_bins, dtype=np.float32)
        idx = int(value // bin_width)
        idx = n_bins - 1 if idx >= n_bins else idx
        features[idx] = 1.0
        return features

    @property
    def num_dims(self):
        return 20 + 8 + 5


class ActionSeqFeature(object):

    def __init__(self, n_dims_action_space, seq_len):
        self._action_seq = [-1] * seq_len
        self._n_dims_action_space = n_dims_action_space

    def reset(self):
        self._action_seq = [-1] * len(self._action_seq)

    def push_action(self, action):
        self._action_seq.pop(0)
        self._action_seq.append(action)

    def features(self):
        features = np.zeros(self._n_dims_action_space * len(self._action_seq),
                            dtype=np.float32)
        for i, action in enumerate(self._action_seq):
            assert action < self._n_dims_action_space
            if action >= 0:
                features[i * self._n_dims_action_space + action] = 1.0
        return features

    @property
    def num_dims(self):
        return self._n_dims_action_space * len(self._action_seq)


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

    @property
    def player_position(self):
        return self.env.player_position
