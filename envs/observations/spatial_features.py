import numpy as np

from envs.common.const import ALLY_TYPE


class UnitType3DFeature(object):

    def __init__(self, type_map, resolution, world_size=(200.0, 176.0)):
        self._type_map = type_map
        self._resolution = resolution
        self._world_size = world_size

    def features(self, observation):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
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
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
        neutral_units = [u for u in observation['units']
                         if u.int_attr.alliance == ALLY_TYPE.NEUTRAL.value]
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
