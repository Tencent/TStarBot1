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


class A2CSimpleAgent(object):

    def __init__(self,
                 dims, 
                 rmsprop_lr=1e-4,
                 rmsprop_eps=1e-8,
                 rollout_num_steps=5,
                 discount=0.99,
                 ent_coef=0,
                 val_coef=1.0,
                 use_gpu=True,
                 init_model_path=None,
                 save_model_dir=None,
                 save_model_freq=10000,
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

        self._actor_critic = FullyConvNet(dims)
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
        ob = torch.from_numpy(np.expand_dims(ob, 0))
        if self._use_gpu: ob = ob.cuda()
        prob_logit, _ = self._actor_critic(Variable(ob))
        action = self._sample_action(prob_logit.data)
        return action.numpy() if not self._use_gpu else action.cpu().numpy()

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
            obs = torch.from_numpy(obs)
            if self._use_gpu: obs = obs.cuda()
            prob_logit, _ = self._actor_critic(Variable(obs, volatile=True))
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
        return obs_mb, action_mb, target_value_mb,  obs

    def _update(self, obs_mb, action_mb, target_value_mb):
        prob_logit, value = self._actor_critic(Variable(torch.cat(obs_mb)))
        log_prob = F.log_softmax(prob_logit, 1)
        prob = F.softmax(prob_logit, 1)
        entropy = -(log_prob * prob).sum(1)

        advantage = Variable(torch.cat(target_value_mb)) - value
        action = Variable(torch.cat(action_mb))
        value_loss = advantage.pow(2).mean() * 0.5
        policy_loss = - (log_prob.gather(1, action) *
                         Variable(advantage.data)).mean()
        entropy_loss = - entropy.mean()
        loss = policy_loss + self._val_coef * value_loss + \
               self._ent_coef * entropy_loss
        print(value_loss, policy_loss, entropy_loss)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self._actor_critic.parameters(), 40)
        self._optimizer.step()

    def _boostrap(self, reward_mb, done_mb, last_obs):
        last_obs = torch.from_numpy(last_obs)
        if self._use_gpu: last_obs = last_obs.cuda()
        _, last_value = self._actor_critic(Variable(last_obs, volatile=True))
        target_value = []
        r = last_value.data.squeeze() * (1 - done_mb[-1])
        for reward, done in reversed(zip(reward_mb, done_mb)):
            r *= 1 - done 
            r = self._discount * r + reward
            target_value.append(r.unsqueeze(1))
        return target_value[::-1]

    def _sample_action(self, logit):
        return F.softmax(Variable(logit), 1).multinomial(1).data

    def _save_model(self, model_path):
        torch.save(self._actor_critic.state_dict(), model_path)

    def _load_model(self, model_path):
        self._actor_critic.load_state_dict(
            torch.load(model_path, map_location=lambda storage, loc: storage))

            
class FullyConvNet(nn.Module):
    def __init__(self, dims):
        super(FullyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5,
                                                     stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                                     stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,
                                                     stride=1, padding=0)
        self.fc = nn.Linear(32 * dims * dims, 64)
        self.value_fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = x.view(x.size(0), -1)
        v = F.relu(self.fc(x1))
        x = self.conv3(x)
        policy = x.view(x.size(0), -1) * 3
        value = self.value_fc(v)
        return policy, value
