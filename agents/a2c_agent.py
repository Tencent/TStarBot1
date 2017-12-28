from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features


class A2CAgent(object):

    def __init__(self,
                 dims, 
                 rmsprop_lr=1e-4,
                 rmsprop_eps=1e-8,
                 rollout_num_steps=5,
                 discount=0.99,
                 ent_coef=1e-3,
                 val_coef=1.0,
                 use_gpu=True):
        self._actor_critic = FullyConvNet(dims)
        if torch.cuda.device_count() > 1:
            self._actor_critic = nn.DataParallel(self._actor_critic)
        if use_gpu:
            self._actor_critic.cuda()
        self._optimizer = optim.RMSprop(self._actor_critic.parameters(),
                                        lr=rmsprop_lr,
                                        eps=rmsprop_eps,
                                        centered=False)
        self._rollout_num_steps = rollout_num_steps
        self._discount = discount
        self._ent_coef = ent_coef
        self._val_coef = val_coef
        self._use_gpu = use_gpu

    def step(self, obs):
        obs_feat = np.eye(5, dtype=np.float32)[np.stack([obs])][:, :, :, 1:]
        obs_feat = torch.from_numpy(obs_feat.transpose((0, 3, 1, 2)))
        if self._use_gpu:
            obs_feat = obs_feat.cuda()
        prob_logit, _ = self._actor_critic(Variable(obs_feat))
        action = self._sample_action(prob_logit.data)
        return action

    def train(self, envs):
        obs, reward, done, _ = envs.reset()
        while True:
            obs_mb, action_mb, reward_mb, obs, reward, done = self._rollout(
                envs, obs, reward, done)
            self._update(obs_mb, action_mb, reward_mb, obs, done)

    def _rollout(self, envs, obs, reward, done):
        num_steps = 0
        obs_mb, action_mb, reward_mb = [], [], []
        while (not done[0]) and num_steps < self._rollout_num_steps:
            obs_feat = np.eye(5, dtype=np.float32)[obs][:, :, :, 1:]
            obs_feat = torch.from_numpy(obs_feat.transpose((0, 3, 1, 2)))
            if self._use_gpu:
                obs_feat = obs_feat.cuda()
            prob_logit, _ = self._actor_critic(Variable(obs_feat))
            action = self._sample_action(prob_logit.data)
            obs_mb.append(obs_feat)
            action_mb.append(action)
            reward_mb.append(torch.cuda.FloatTensor(reward) if self._use_gpu
                             else torch.Tensor(reward))
            if self._use_gpu:
                action = action.cpu()
            obs, reward, done, _ = envs.step(action.numpy())
            num_steps += 1
        return obs_mb, action_mb, reward_mb, obs, reward, done

    def _update(self, obs_mb, action_mb, reward_mb, obs, done):
        prob_logit, value = self._actor_critic(Variable(torch.cat(obs_mb, 0)))
        log_prob, prob = F.log_softmax(prob_logit, 1), F.softmax(prob_logit, 1)
        entropy = -(log_prob * prob).sum(1)

        r = value.data.new(len(done), 1).zero_()
        if not done[0]:
            last_obs_feat = np.eye(5, dtype=np.float32)[obs][:, :, :, 1:]
            last_obs_feat = torch.from_numpy(
                last_obs_feat.transpose((0, 3, 1, 2)))
            if self._use_gpu:
                last_obs_feat = last_obs_feat.cuda()
            _, last_value = self._actor_critic(Variable(last_obs_feat))
            r = last_value.data
        returns = []
        for reward in reversed(reward_mb):
            r = self._discount * r + reward.unsqueeze(1)
            returns.append(r)
        returns = Variable(torch.cat(returns[::-1]))

        advantage = returns - value
        action = Variable(torch.cat(action_mb))
        value_loss = advantage.pow(2).mean() * 0.5
        policy_loss = - (log_prob.gather(1, action) *
                         Variable(advantage.data)).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + self._val_coef * value_loss + \
               self._ent_coef * entropy_loss
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self._actor_critic.parameters(), 40)
        self._optimizer.step()

    def _sample_action(self, logit):
        noise = torch.log(-torch.log(logit.new(logit.size()).uniform_()))
        _, action = torch.max(logit - noise, 1)
        return action.unsqueeze(1)

            
class FullyConvNet(nn.Module):
    def __init__(self, dims):
        super(FullyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5,
                                                     stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                                     stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,
                                                     stride=1, padding=0)
        self.fc = nn.Linear(32 * dims * dims, 128)
        self.value_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = x.view(x.size(0), -1)
        v = F.relu(self.fc(x1))
        x = self.conv3(x)
        policy = x.view(x.size(0), -1) * 3
        value = self.value_fc(v)
        return policy, value
