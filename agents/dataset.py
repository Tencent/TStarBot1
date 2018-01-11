from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import StringIO
import tarfile
import gzip
import cPickle as pickle

import numpy as np
from torch.utils.data import Dataset

from pysc2.lib.actions import FUNCTIONS
from pysc2.lib.actions import TYPES
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType


class SCReplayDataset(Dataset):
    """StarCraftII replay dataset."""

    def __init__(self, root_dir, resolution, transform=None):
        self._transform = transform
        self._resolution = resolution
        self._init_filelist(root_dir)
        self._init_id_mapper()
        self._init_action_spec()
        self._init_observation_spec()

    @property
    def action_spec(self):
        return (self._action_head_dims,
                self._action_args_map)

    @property
    def observation_spec(self):
        return (self._num_channels_screen,
                self._num_channels_minimap,
                self._resolution)

    def __len__(self):
        return self._total_num_frames

    def __getitem__(self, idx):
        file_id, frame_id = self._map_global_id_to_local(idx)
        data = self._load_data(self._filelist[file_id], frame_id)
        obs_screen, obs_minimap, action_available = self._transform_observation(
            data["observation"])
        policy_label = self._transform_actions(data["actions"], action_available)
        value_label = 1 if data["result"] == 1 else 0
        sample = {'screen_feature': obs_screen,
                  'minimap_feature': obs_minimap,
                  'action_available': action_available,
                  'policy_label': policy_label,
                  'value_label': value_label}
        if self._transform:
            sample = self._transform(sample)
        return sample

    def _init_filelist(self, root_dir):
        self._filelist = [os.path.join(root_dir, filename)
                          for filename in os.listdir(root_dir)
                          if filename.endswith(".tar")]
        self._num_frames_list = [int(filename[filename.rfind('-')+1:-4])
                                 for filename in self._filelist
                                 if filename.endswith(".tar")]
        self._total_num_frames = sum(self._num_frames_list)

    def _transform_actions(self, actions, action_available):
        label = np.zeros(self._action_total_dim, dtype=np.float32)
        if len(actions) > 0:
            action = actions[0]
        has_valid_action = False
        for action in actions:
            if action_available[action.function] == 0: # 0 valid, 1e30 invalid
                label[action.function] = 1
                for arg_id, arg_val in zip(self._action_args_map[action.function],
                                           action.arguments):
                    if len(arg_val) == 1:
                        arg_val = arg_val[0]
                        assert arg_val < self._action_head_dims[arg_id + 1]
                        label[self._action_args_offset[arg_id] + arg_val] = 1
                    elif len(arg_val) == 2:
                        arg_val = arg_val[1] * self._resolution + arg_val[0]
                        assert arg_val < self._action_head_dims[arg_id + 1]
                        label[self._action_args_offset[arg_id] + arg_val] = 1
                    else:
                        raise NotImplementedError
                has_valid_action = True
                break
        if not has_valid_action:
            label[0] = 1
        return label

    def _init_action_spec(self):
        self._action_head_dims = [len(FUNCTIONS)]
        for argument in TYPES:
            if len(argument.sizes) == 2:
                self._action_head_dims.append(self._resolution ** 2)
            elif len(argument.sizes) == 1:
                self._action_head_dims.append(argument.sizes[0])
            else:
                raise NotImplementedError
        self._action_args_map = []
        for func in FUNCTIONS:
            self._action_args_map.append([arg.id for arg in func.args])
        self._action_args_offset = [sum(self._action_head_dims[:i+1])
            for i in xrange(len(self._action_head_dims) - 1)]
        self._action_total_dim = sum(self._action_head_dims)

    def _init_observation_spec(self):
        def get_spatial_channels(specs):
            num_channels = 0
            for spec in specs:
                if spec.type == FeatureType.CATEGORICAL:
                    num_channels += spec.scale - 1
                else:
                    num_channels += 1
            return num_channels
        self._num_channels_screen = get_spatial_channels(SCREEN_FEATURES)
        self._num_channels_minimap = get_spatial_channels(MINIMAP_FEATURES)

    def _transform_observation(self, observation):
        obs_screen = self._transform_spatial_features(
            observation["screen"], SCREEN_FEATURES)
        obs_minimap = self._transform_spatial_features(
            observation["minimap"], MINIMAP_FEATURES)
        num_actions =  self._action_head_dims[0]
        action_available = np.ones(num_actions, dtype=np.float32) * 1e30
        action_available[observation["available_actions"]] = 0
        assert obs_screen.shape[1] == self._resolution
        assert obs_screen.shape[2] == self._resolution
        assert obs_minimap.shape[1] == self._resolution
        assert obs_minimap.shape[2] == self._resolution
        return obs_screen, obs_minimap, action_available

    def _transform_spatial_features(self, obs, specs):
        features = []
        for ob, spec in zip(obs, specs):
            if spec.type == FeatureType.CATEGORICAL: features.append(
                    np.eye(spec.scale, dtype=np.float32)[ob][:, :, 1:])
            else:
                features.append(
                    np.expand_dims(np.log(ob + 1, dtype=np.float32), axis=2))
        return np.transpose(np.concatenate(features, axis=2), (2, 0, 1))

    def _load_data(self, filepath, frame_id):
        with tarfile.open(filepath, 'r') as tar:
            tarinfo = tar.getmembers()[frame_id]
            f = tar.extractfile(tarinfo)
            content = f.read()
            gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
            data = pickle.loads(gfile.read())
        return data

    def _init_id_mapper(self):
        self._sum_frames_list = [0] * len(self._num_frames_list)
        self._sum_frames_list[0] = self._num_frames_list[0]
        for i in xrange(1, len(self._num_frames_list)):
            self._sum_frames_list[i] = self._sum_frames_list[i - 1] + \
                                       self._num_frames_list[i]

    def _map_global_id_to_local(self, idx):
        # binary search 
        assert idx < self._sum_frames_list[-1]
        l, r = 0, len(self._sum_frames_list)
        while l < r:
            mid = (l + r) // 2
            if self._sum_frames_list[mid] >= idx + 1:
                r = mid
            else:
                l = mid + 1
        file_id = r
        frame_id = idx - self._sum_frames_list[file_id - 1] \
            if file_id > 0 else idx
        return file_id, frame_id 
