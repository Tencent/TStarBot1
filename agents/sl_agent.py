from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pysc2.lib import actions
from pysc2.lib import features


class SLAgent(object):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 batch_size=64,
                 num_dataloader_worker=8,
                 rmsprop_lr=1e-4,
                 rmsprop_eps=1e-8,
                 use_gpu=True,
                 init_model_path=None,
                 save_model_dir=None,
                 save_model_freq=10000,
                 seed=0):
        self._batch_size = batch_size
        self._num_dataloader_worker = num_dataloader_worker
        self._action_dims, = action_spec
        self._use_gpu = use_gpu
        self._save_model_dir = save_model_dir
        self._save_model_freq = save_model_freq
        self._steps_count = 0

        torch.manual_seed(seed)
        if use_gpu: torch.cuda.manual_seed(seed)

        in_channels_screen, in_channels_minimap, resolution = observation_spec
        self._actor_critic = self._create_model(
            self._action_dims, in_channels_screen, in_channels_minimap,
            resolution, init_model_path, use_gpu)
        
        self._optimizer = optim.RMSprop(
            self._actor_critic.parameters(), lr=rmsprop_lr,
            eps=rmsprop_eps, centered=False)

        if init_model_path:
            self._steps_count = int(init_model_path[
                init_model_path.rfind('-')+1:])

    def step(self, ob):
        raise NotImplementedError

    def train(self, dataset_train, dataset_dev):
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=self._batch_size,
                                      shuffle=True,
                                      pin_memory=self._use_gpu,
                                      num_workers=self._num_dataloader_worker)
        dataloader_dev = DataLoader(dataset_dev,
                                    batch_size=self._batch_size,
                                    shuffle=False,
                                    pin_memory=self._use_gpu,
                                    num_workers=self._num_dataloader_worker)
        while True:
            for batch in dataloader_train:
                screen_feature = batch["screen_feature"]
                minimap_feature = batch["minimap_feature"]
                action_available = batch["action_available"]
                policy_label = batch["policy_label"]
                value_label = batch["value_label"]
                if self._use_gpu:
                    screen_feature = screen_feature.cuda()
                    minimap_feature = minimap_feature.cuda()
                    action_available = action_available.cuda()
                    policy_label = policy_label.cuda()
                    value_label = value_label.cuda()

                policy_logprob, value_logit = self._actor_critic(
                    screen=Variable(screen_feature),
                    minimap=Variable(minimap_feature),
                    mask=Variable(action_available))
                print(policy_logprob.size(), value_logit.size())
                policy_cross_ent = (-log_policy * Variable(policy_label)).sum(1)
                policy_loss = policy_cross_ent.mean()
                value_loss = F.binary_cross_entropy_with_logits(
                    value_logit, Variable(value_label))
                loss = policy_loss + value_loss

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._steps_count += 1
                if self._steps_count % self._save_model_freq == 0:
                    self._save_model(os.path.join(
                        self._save_model_dir,
                        'agent.model-%d' % self._steps_count))

    def _create_model(self, action_dims, in_channels_screen,
                      in_channels_minimap, resolution, init_model_path,
                      use_gpu):
        model = FullyConvNet(
            resolution=resolution,
            in_channels_screen=in_channels_screen,
            in_channels_minimap=in_channels_minimap,
            out_channels_spatial=3,
            out_dims_nonspatial=action_dims[0:1] + action_dims[4:])
        model.apply(weights_init)
        if init_model_path:
            self._load_model(init_model_path)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if use_gpu:
            model.cuda()
        return model

    def _save_model(self, model_path):
        torch.save(self._actor_critic.state_dict(), model_path)

    def _load_model(self, model_path):
        self._actor_critic.load_state_dict(
            torch.load(model_path, map_location=lambda storage, loc: storage))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

            
class FullyConvNet(nn.Module):
    def __init__(self,
                 resolution,
                 in_channels_screen,
                 in_channels_minimap,
                 out_channels_spatial,
                 out_dims_nonspatial):
        super(FullyConvNet, self).__init__()
        self.screen_conv1 = nn.Conv2d(in_channels=in_channels_screen,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2)
        self.screen_conv2 = nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.minimap_conv1 = nn.Conv2d(in_channels=in_channels_minimap,
                                       out_channels=16,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)
        self.minimap_conv2 = nn.Conv2d(in_channels=16,
                                       out_channels=32,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.spatial_policy_conv = nn.Conv2d(in_channels=64,
                                             out_channels=out_channels_spatial,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
        self.state_fc = nn.Linear(64 * (resolution ** 2), 256)
        self.value_fc = nn.Linear(256, 1)
        self.nonspatial_policy_fc = nn.Linear(256, sum(out_dims_nonspatial))
        self._out_dims_nonspatial = out_dims_nonspatial

    def forward(self, screen, minimap, mask):
        screen = F.relu(self.screen_conv1(screen))
        screen = F.relu(self.screen_conv2(screen))
        minimap = F.relu(self.minimap_conv1(minimap))
        minimap = F.relu(self.minimap_conv2(minimap))
        screen_minimap = torch.cat((screen, minimap), 1)
        state = F.relu(self.state_fc(
            screen_minimap.view(screen_minimap.size(0), -1)))

        spatial_policy = self.spatial_policy_conv(screen_minimap)
        spatial_policy1 = spatial_policy.view(spatial_policy.size(0), -1)
        spatial_policy = torch.cat(
            [chunk.contiguous().view(chunk.size(0), -1)
             for chunk in spatial_policy.chunk(spatial_policy.size(1), dim=1)],
            dim=1)
        print((spatial_policy - spatial_policy1).pow(2).mean())
        nonspatial_policy = self.nonspatial_policy_fc(state)

        value_logit = self.value_fc(state)
        first_dim = self._out_dims_nonspatial[0] 
        print(nonspatial_policy.size(), first_dim, nonspatial_policy[:, :first_dim].size())
        policy_logit = torch.cat([nonspatial_policy[:, :first_dim] * mask,
                                  spatial_policy,
                                  nonspatial_policy[first_dim:]],
                                 dim=0)
        policy_logprob = self._group_log_softmax(
            policy_logit, self._out_dims_nonspatial)
        return policy_logprob, value_logit

    def _group_log_softmax(x, group_dims):
        idx = 0
        for dim in group_dims:
            x[idx:idx+dim] = F.log_softmax(x[idx:idx+dim], dim=1)
            idx += dim
        return x
