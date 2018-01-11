from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
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
                 use_gpu=True,
                 init_model_path=None,
                 seed=0):
        self._batch_size = batch_size
        self._action_dims, = action_spec
        self._use_gpu = use_gpu

        torch.manual_seed(seed)
        if use_gpu: torch.cuda.manual_seed(seed)

        in_channels_screen, in_channels_minimap, resolution = observation_spec
        self._actor_critic = self._create_model(
            self._action_dims, in_channels_screen, in_channels_minimap,
            resolution, init_model_path, use_gpu)
        

    def step(self, ob):
        raise NotImplementedError

    def train(self,
              dataset_train,
              dataset_dev,
              learning_rate,
              num_dataloader_worker=8,
              save_model_dir=None,
              save_model_freq=100000,
              print_freq=1000,
              max_epochs=10000):
        optimizer = optim.RMSprop(self._actor_critic.parameters(),
                                  lr=learning_rate,
                                  eps=1e-5,
                                  centered=False)

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=self._batch_size,
                                      shuffle=True,
                                      pin_memory=self._use_gpu,
                                      num_workers=num_dataloader_worker)
        dataloader_dev = DataLoader(dataset_dev,
                                    batch_size=self._batch_size,
                                    shuffle=False,
                                    pin_memory=self._use_gpu,
                                    num_workers=num_dataloader_worker)
        num_batches, num_epochs, total_loss = 0, 0, 0
        last_time = time.time()
        while num_epochs < max_epochs:
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
                policy_cross_ent = -policy_logprob * Variable(policy_label)
                policy_loss = policy_cross_ent.sum(1).mean()
                value_loss = F.binary_cross_entropy_with_logits(
                    value_logit.squeeze(1), Variable(value_label.float()))
                loss = policy_loss + value_loss
                total_loss += loss[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_batches += 1
                if num_batches % print_freq == 0:
                    print("Training Epochs: %d  Batches: %d Avg Train Loss: %f"
                          " Speed: %.2f s/batch"
                          % (num_epochs, num_batches, total_loss / print_freq,
                             (time.time() - last_time) / print_freq))
                    last_time = time.time()
                    total_loss = 0
                if num_batches % save_model_freq == 0:
                    valid_loss = self.evaluate(dataloader_dev)
                    print("Training Epochs: %d  Avg Valid Loss: %f"
                          % (num_epochs, valid_loss))
                    self._save_model(os.path.join(
                        save_model_dir, 'agent.model-%d' % num_batches))
            num_epochs += 1

    def evaluate(self, dataloader_dev):
        num_batches, total_loss = 0, 0
        for batch in dataloader_dev:
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
                screen=Variable(screen_feature, volatile=True),
                minimap=Variable(minimap_feature, volatile=True),
                mask=Variable(action_available, volatile=True))
            policy_cross_ent = -policy_logprob * Variable(policy_label,
                                                          volatile=True)
            policy_loss = policy_cross_ent.sum(1).mean()
            value_loss = F.binary_cross_entropy_with_logits(
                value_logit.squeeze(1), Variable(value_label.float(),
                                                 volatile=True))
            loss = policy_loss + value_loss
            total_loss += loss[0]
            num_batches += 1
        return total_loss / num_batches

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
        self._action_dims = out_dims_nonspatial[0:1] + [resolution ** 2] * 3 \
                            + out_dims_nonspatial[1:]

    def forward(self, screen, minimap, mask):
        screen = F.relu(self.screen_conv1(screen))
        screen = F.relu(self.screen_conv2(screen))
        minimap = F.relu(self.minimap_conv1(minimap))
        minimap = F.relu(self.minimap_conv2(minimap))
        screen_minimap = torch.cat((screen, minimap), 1)
        state = F.relu(self.state_fc(
            screen_minimap.view(screen_minimap.size(0), -1)))

        spatial_policy = self.spatial_policy_conv(screen_minimap)
        spatial_policy = spatial_policy.view(spatial_policy.size(0), -1)
        nonspatial_policy = self.nonspatial_policy_fc(state)

        value_logit = self.value_fc(state)
        first_dim = self._action_dims[0] 
        policy_logit = torch.cat([nonspatial_policy[:, :first_dim] - mask,
                                  spatial_policy,
                                  nonspatial_policy[:, first_dim:]],
                                 dim=1)
        policy_logprob = self._group_log_softmax(
            policy_logit, self._action_dims)
        return policy_logprob, value_logit

    def _group_log_softmax(self, x, group_dims):
        idx = 0
        output = []
        for dim in group_dims:
            output.append(F.log_softmax(x[:, idx:idx+dim], dim=1))
            idx += dim
        return torch.cat(output, dim=1)
