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
                 use_gpu=True,
                 init_model_path=None,
                 seed=0):
        self._action_dims, self._action_args_map = action_spec
        self._use_gpu = use_gpu

        torch.manual_seed(seed)
        if use_gpu: torch.cuda.manual_seed(seed)

        in_channels_screen, in_channels_minimap, resolution = observation_spec
        self._actor_critic = self._create_model(
            self._action_dims, in_channels_screen, in_channels_minimap,
            resolution, init_model_path, use_gpu)
        
    def step(self, ob, info):
        screen_feature = torch.from_numpy(np.expand_dims(ob[0], 0))
        minimap_feature = torch.from_numpy(np.expand_dims(ob[1], 0))
        mask = np.ones((1, self._action_dims[0]), dtype=np.float32) * 1e30
        mask[0, info] = 0
        mask = torch.from_numpy(mask)
        if self._use_gpu:
            screen_feature = screen_feature.cuda()
            minimap_feature = minimap_feature.cuda()
            mask = mask.cuda()
        policy_logprob, value = self._actor_critic(
            screen=Variable(screen_feature, volatile=True),
            minimap=Variable(minimap_feature, volatile=True),
            mask=Variable(mask, volatile=True))
        # value
        victory_prob = value.data[0, 0]
        # control - function id
        function_id = torch.max(
            policy_logprob[:, :self._action_dims[0]], 1)[1].data[0]
        # control - function arguments
        arguments = []
        for arg_id in self._action_args_map[function_id]:
            l = sum(self._action_dims[:arg_id+1])
            r = sum(self._action_dims[:arg_id+2])
            arg_val = torch.max(policy_logprob[:, l:r], 1)[1].data[0]
            arguments.append(arg_val)
        print("Function ID: %d, Arguments: %s, Winning Probability: %f"
              % (function_id, arguments, victory_prob))
        return [function_id] + arguments

    def train(self,
              dataset_train,
              dataset_dev,
              learning_rate=1e-4,
              optimizer_type="adam",
              batch_size=64,
              num_dataloader_worker=8,
              save_model_dir=None,
              save_model_freq=100000,
              print_freq=1000,
              max_sampled_dev_ins=3200,
              max_epochs=10000):
        if optimizer_type == "adam":
            optimizer = optim.Adam(self._actor_critic.parameters(),
                                   lr=learning_rate)
        elif optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(self._actor_critic.parameters(),
                                      lr=learning_rate,
                                      eps=1e-5,
                                      centered=False)
        else:
            raise NotImplementedError

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=self._use_gpu,
                                      num_workers=num_dataloader_worker)
        dataloader_dev = DataLoader(dataset_dev,
                                    batch_size=batch_size,
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

                policy_logprob, value = self._actor_critic(
                    screen=Variable(screen_feature),
                    minimap=Variable(minimap_feature),
                    mask=Variable(action_available))
                policy_cross_ent = -policy_logprob * Variable(policy_label)
                policy_loss = policy_cross_ent.sum(1).mean()
                value_loss = F.binary_cross_entropy(
                    value.squeeze(1), Variable(value_label.float()))
                loss = policy_loss + value_loss
                total_loss += loss[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_batches += 1
                if num_batches % print_freq == 0:
                    print("Epochs: %d Batches: %d Avg Train Loss: %f "
                          "Speed: %.2f s/batch"
                          % (num_epochs, num_batches, total_loss / print_freq,
                             (time.time() - last_time) / print_freq))
                    last_time = time.time()
                    total_loss = 0
                if num_batches % save_model_freq == 0:
                    valid_loss, value_acc, action_acc = self.evaluate(
                        dataloader_dev, max_sampled_dev_ins)
                    print("Epochs: %d Validation Loss: %f Value Accuracy: %f "
                          "Action Accuracy: %f"
                          % (num_epochs, valid_loss, value_acc, action_acc))
                    self._save_model(os.path.join(
                        save_model_dir, 'agent.model-%d' % num_batches))
            num_epochs += 1

    def evaluate(self, dataloader_dev, max_instances=None):
        num_instances, total_loss = 0, 0
        correct_value = 0
        correct_action, correct_screen, correct_minimap = 0, 0, 0
        for batch in dataloader_dev:
            if max_instances and max_instances <= num_instances:
                break
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

            policy_logprob, value = self._actor_critic(
                screen=Variable(screen_feature, volatile=True),
                minimap=Variable(minimap_feature, volatile=True),
                mask=Variable(action_available, volatile=True))
            # loss
            policy_cross_ent = -policy_logprob * Variable(policy_label,
                                                          volatile=True)
            policy_loss = policy_cross_ent.sum(1).mean()
            value_loss = F.binary_cross_entropy(
                value.squeeze(1), Variable(value_label.float(), volatile=True))
            loss = policy_loss + value_loss
            total_loss += loss[0]
            # value accuracy
            correct_value += ((value.squeeze(1) > 0.5).long() == Variable(
                value_label, volatile=True)).sum().data[0]
            # action accuracy
            l, r = 0, self._action_dims[0]
            _, predicted = torch.max(policy_logprob[:, l:r], 1)
            _, label = torch.max(
                Variable(policy_label[:, l:r], volatile=True), 1)
            correct_action += (predicted == label).sum().data[0]
            num_instances += value_label.size(0)

        return (total_loss / num_instances,
                correct_value / float(num_instances),
                correct_action / float(num_instances))

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
            model.load_state_dict(
                torch.load(init_model_path,
                           map_location=lambda storage, loc: storage))

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if use_gpu:
            model.cuda()
        return model

    def _save_model(self, model_path):
        torch.save(self._actor_critic.state_dict(), model_path)


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

        value = F.sigmoid(self.value_fc(state))
        first_dim = self._action_dims[0] 
        policy_logit = torch.cat([nonspatial_policy[:, :first_dim] - mask,
                                  spatial_policy,
                                  nonspatial_policy[:, first_dim:]],
                                 dim=1)
        policy_logprob = self._group_log_softmax(
            policy_logit, self._action_dims)
        return policy_logprob, value

    def _group_log_softmax(self, x, group_dims):
        idx = 0
        output = []
        for dim in group_dims:
            output.append(F.log_softmax(x[:, idx:idx+dim], dim=1))
            idx += dim
        return torch.cat(output, dim=1)
