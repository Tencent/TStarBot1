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

from pysc2.lib import actions
from pysc2.lib import features


class A2CScriptedAgent(object):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 rmsprop_lr=1e-4,
                 rmsprop_eps=1e-8,
                 rollout_num_steps=5,
                 discount=0.999,
                 ent_coef=1e-2,
                 val_coef=0.5,
                 use_gpu=True,
                 init_model_path=None,
                 save_model_dir=None,
                 save_model_freq=10000,
                 enable_batchnorm=False,
                 seed=0):
        self._rollout_num_steps = rollout_num_steps
        self._discount = discount
        self._ent_coef = ent_coef
        self._val_coef = val_coef
        self._use_gpu = use_gpu
        self._save_model_dir = save_model_dir
        self._save_model_freq = save_model_freq
        self._steps_count = 0

        torch.manual_seed(seed)
        if use_gpu: torch.cuda.manual_seed(seed)

        self._actor_critic = FullyConvNet(
            resolution=observation_spec[2],
            in_channels_screen=observation_spec[0],
            in_channels_minimap=observation_spec[1],
            out_dims=action_spec[0],
            enable_batchnorm=enable_batchnorm)
        self._actor_critic.apply(weights_init)
        if init_model_path:
            self._load_model(init_model_path)
            self._steps_count = int(init_model_path[
                init_model_path.rfind('-')+1:])

        if torch.cuda.device_count() > 1:
            self._actor_critic = nn.DataParallel(self._actor_critic)
        if use_gpu:
            self._actor_critic.cuda()
        self._optimizer = optim.RMSprop(
            self._actor_critic.parameters(), lr=rmsprop_lr,
            eps=rmsprop_eps, centered=False)

    def step(self, ob):
        if isinstance(ob, tuple):
            ob = tuple(np.expand_dims(ob, 0) for o in ob)
        else:
            ob = np.expand_dims(ob, 0)
        ob = self._ndarray_to_tensor(ob)
        prob_logit, _ = self._actor_critic(
            tuple(Variable(tensor, volatile=True) for tensor in ob))
        action = self._sample_action(prob_logit.data)
        return action.numpy()[0] if not self._use_gpu else action.cpu().numpy()

    def train(self, envs):
        obs, _ = envs.reset()
        while True:
            obs_mb, action_mb, target_value_mb, obs = self._rollout(envs, obs)
            self._update(obs_mb, action_mb, target_value_mb)
            self._steps_count += 1
            if self._steps_count % self._save_model_freq == 0:
                self._save_model(os.path.join(
                    self._save_model_dir, 'agent.model-%d' % self._steps_count))

    def _rollout(self, envs, obs):
        obs_mb, action_mb, reward_mb, done_mb = [], [], [], []
        for _ in xrange(self._rollout_num_steps):
            obs = self._ndarray_to_tensor(obs)
            prob_logit, _ = self._actor_critic(
                tuple(Variable(tensor, volatile=True) for tensor in obs))
            action = self._sample_action(prob_logit.data)
            obs_mb.append(obs)
            action_mb.append(action)
            obs, reward, done, _ = envs.step(action.numpy() if not self._use_gpu
                                             else action.cpu().numpy())
            reward_mb.append(torch.Tensor(reward) if not self._use_gpu
                             else torch.cuda.FloatTensor(reward))
            done_mb.append(torch.Tensor(done.tolist()) if not self._use_gpu
                           else torch.cuda.FloatTensor(done.tolist()))
        target_value_mb = self._boostrap(reward_mb, done_mb, obs)
        return obs_mb, action_mb, target_value_mb, obs

    def _update(self, obs_mb, action_mb, target_value_mb):
        logit, value = self._actor_critic(
            tuple(Variable(torch.cat([obs[c] for obs in obs_mb])) 
                  for c in xrange(len(obs_mb[0]))))
        log_prob = F.log_softmax(logit, 1)
        prob = F.softmax(logit, 1)
        entropy = -(log_prob * prob).sum(1)

        advantage = Variable(torch.cat(target_value_mb)) - value
        action = Variable(torch.cat(action_mb))
        value_loss = advantage.pow(2).mean() * 0.5
        policy_loss = - (log_prob.gather(1, action) *
                         Variable(advantage.data)).mean()
        entropy_loss = - entropy.mean()
        #print(policy_loss, value_loss, entropy_loss)
        loss = policy_loss + self._val_coef * value_loss + \
               self._ent_coef * entropy_loss
        #print(entropy_loss)

        self._optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(self._actor_critic.parameters(), 40)
        self._optimizer.step()

    def _boostrap(self, reward_mb, done_mb, last_obs):
        last_obs = self._ndarray_to_tensor(last_obs)
        _, last_value = self._actor_critic(
            tuple(Variable(tensor, volatile=True) for tensor in last_obs))
        target_value = []
        r = last_value.data.squeeze() * (1 - done_mb[-1])
        for reward, done in reversed(zip(reward_mb, done_mb)):
            r *= 1 - done 
            r = self._discount * r + reward
            target_value.append(r.unsqueeze(1))
        return target_value[::-1]

    def _ndarray_to_tensor(self, arrays):
        if isinstance(arrays, tuple):
            if self._use_gpu:
                return [torch.from_numpy(array).cuda() for array in arrays]
            else:
                return [torch.from_numpy(array) for array in arrays]
        else:
            if self._use_gpu:
                return torch.from_numpy(arrays).cuda()
            else:
                return torch.from_numpy(arrays)

    def _sample_action(self, logit):
        return F.softmax(Variable(logit), 1).multinomial(1).data

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
                 out_dims,
                 enable_batchnorm=False):
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
        if enable_batchnorm:
            self.screen_bn1 = nn.BatchNorm2d(16)
            self.screen_bn2 = nn.BatchNorm2d(32)
            self.minimap_bn1 = nn.BatchNorm2d(16)
            self.minimap_bn2 = nn.BatchNorm2d(32)
            self.player_bn = nn.BatchNorm2d(10)
            self.state_bn = nn.BatchNorm1d(256)
        self.state_fc = nn.Linear(74 * (resolution ** 2), 256)
        self.value_fc = nn.Linear(256, 1)
        self.policy_fc = nn.Linear(256, out_dims)
        self._enable_batchnorm = enable_batchnorm

    def forward(self, x):
        screen, minimap, player = x
        player = player.clone().repeat(
            screen.size(2), screen.size(3), 1, 1).permute(2, 3, 0, 1)
        if self._enable_batchnorm:
            screen = F.leaky_relu(self.screen_bn1(self.screen_conv1(screen)))
            screen = F.leaky_relu(self.screen_bn2(self.screen_conv2(screen)))
            minimap = F.leaky_relu(self.minimap_bn1(self.minimap_conv1(minimap)))
            minimap = F.leaky_relu(self.minimap_bn2(self.minimap_conv2(minimap)))
            player = self.player_bn(player.contiguous())
        else:
            screen = F.leaky_relu(self.screen_conv1(screen))
            screen = F.leaky_relu(self.screen_conv2(screen))
            minimap = F.leaky_relu(self.minimap_conv1(minimap))
            minimap = F.leaky_relu(self.minimap_conv2(minimap))
        screen_minimap = torch.cat((screen, minimap, player), 1)
        if self._enable_batchnorm:
            state = F.leaky_relu(self.state_bn(self.state_fc(
                screen_minimap.view(screen_minimap.size(0), -1))))
        else:
            state = F.leaky_relu(self.state_fc(
                screen_minimap.view(screen_minimap.size(0), -1)))
        value = self.value_fc(state)
        policy_logit = self.policy_fc(state)
        return policy_logit, value
