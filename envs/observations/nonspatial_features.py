import numpy as np

from envs.common.const import ALLY_TYPE


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
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
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
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
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
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        self_flying_units = [u for u in self_units if u.bool_attr.is_flying]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
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
