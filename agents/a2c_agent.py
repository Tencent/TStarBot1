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


class A2CAgent(object):

    def __init__(self,
                 dims, 
                 observation_spec,
                 action_spec,
                 rmsprop_lr=1e-4,
                 rmsprop_eps=1e-8,
                 rollout_num_steps=5,
                 discount=0.99,
                 ent_coef=1e-3,
                 val_coef=1.0,
                 func_id_loss_coef=0.01,
                 use_gpu=True,
                 init_model_path=None,
                 save_model_dir=None,
                 save_model_freq=500,
                 seed=1):
        self._rollout_num_steps = rollout_num_steps
        self._discount = discount
        self._ent_coef = ent_coef
        self._val_coef = val_coef
        self._use_gpu = use_gpu
        self._action_num_heads = action_spec[0]
        self._action_head_sizes = action_spec[1]
        self._action_args_map = action_spec[2]
        self._save_model_dir = save_model_dir
        self._save_model_freq = save_model_freq
        self._func_id_loss_coef = func_id_loss_coef

        torch.manual_seed(seed)
        if use_gpu: torch.cuda.manual_seed(seed)

        self._actor_critic = FullyConvNet(
            screen_minimap_size=dims,
            in_channels_screen=observation_spec[0],
            in_channels_minimap=observation_spec[1],
            out_channels_spatial=3,
            out_dims_nonspatial= [self._action_num_heads] + \
                                 self._action_head_sizes[3:])
        self._actor_critic.apply(weights_init)
        if init_model_path:
            self._load_model(init_model_path)

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
        return action.numpy() if not self._use_gpu else action.cpu().numpy()

    def train(self, envs):
        obs, infos = envs.reset()
        steps = 0
        while True:
            obs_mb, action_mb, target_value_mb, obs, infos = self._rollout(
                envs, obs, infos)
            self._update(obs_mb, action_mb, target_value_mb)
            steps += 1
            if steps % self._save_model_freq == 0:
                self._save_model(os.path.join(self._save_model_dir,
                                              'agent.model-%d' % steps))

    def _rollout(self, envs, obs, infos):
        obs_mb, action_mb, reward_mb, done_mb = [], [], [], []
        for _ in xrange(self._rollout_num_steps):
            obs = self._ndarray_to_tensor(obs)
            prob_logit, _ = self._actor_critic(
                tuple(Variable(tensor, volatile=True) for tensor in obs))
            action = self._sample_action(prob_logit, infos)
            obs_mb.append(obs)
            action_mb += action
            obs, reward, done, infos = envs.step(action)
            reward_mb.append(torch.Tensor(reward) if not self._use_gpu
                             else torch.cuda.FloatTensor(reward))
            done_mb.append(torch.Tensor(done.tolist()) if not self._use_gpu
                           else torch.cuda.FloatTensor(done.tolist()))
        target_value_mb = self._boostrap(reward_mb, done_mb, obs)
        return obs_mb, action_mb, target_value_mb, obs, infos

    def _update(self, obs_mb, action_mb, target_value_mb):
        prob_logit, value = self._actor_critic(
            tuple(Variable(torch.cat([obs[c] for obs in obs_mb])) 
                  for c in xrange(len(obs_mb[0]))))
        advantage = Variable(torch.cat(target_value_mb)) - value
        func_logit = prob_logit[0]
        func_action = torch.LongTensor([[action[0]] for action in action_mb])
        if self._use_gpu:
            func_action = func_action.cuda()
        func_log_prob = F.log_softmax(func_logit, 1)
        func_prob = F.softmax(func_logit, 1)
        entropy = -(func_log_prob * func_prob).sum(1)
        log_prob_action = func_log_prob.gather(1, Variable(func_action))
        entropy = entropy * self._func_id_loss_coef
        log_prob_action = log_prob_action * self._func_id_loss_coef

        for idx, action in enumerate(action_mb):
            func = action[0]
            for arg_val, arg_id in zip(action[1:], self._action_args_map[func]):
                if arg_id == 3: # queued is forced to False
                    continue
                arg_logit = prob_logit[arg_id + 1][idx, :]
                log_prob = F.log_softmax(arg_logit, 0)
                prob = F.softmax(arg_logit, 0)
                entropy[idx] = entropy[idx] - (log_prob * prob).sum()
                log_prob_action[idx] = log_prob_action[idx] + log_prob[arg_val]

        value_loss = advantage.pow(2).mean() * 0.5
        policy_loss = - (log_prob_action * Variable(advantage.data)).mean()
        entropy_loss = - entropy.mean()
        loss = policy_loss + self._val_coef * value_loss + \
            self._ent_coef * entropy_loss

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self._actor_critic.parameters(), 40)
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

    def _sample_action(self, logits, available_actions):
        # sample function ids
        function_logit = logits[0]
        mask = function_logit.data.new(function_logit.size()).zero_()
        for i, valids in enumerate(available_actions):
            mask[[i], valids] = 1
        probs = F.softmax(logits[0], 1) + 1e-20
        masked_probs = probs * Variable(mask)
        functions = masked_probs.multinomial(1).data
        # sample arguments
        actions = []
        for i in range(functions.size(0)):
            func = functions[i, 0]
            cur_action = [func]
            for j in self._action_args_map[func]:
                if j == 3: # queued is forced to False
                    cur_action.append(0)
                    continue
                arg_logit = logits[j + 1]
                arg = F.softmax(arg_logit[i, :], 0).multinomial(1).data[0]
                cur_action.append(arg)
            actions.append(cur_action)
        return actions

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
                 screen_minimap_size,
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
        self.fc = nn.Linear(64 * (screen_minimap_size ** 2), 256)
        self.nonspatial_policy_fc = nn.Linear(256, sum(out_dims_nonspatial))
        self.value_fc = nn.Linear(256, 1)
        self._out_dims = out_dims_nonspatial

    def forward(self, x):
        screen_x, minimap_x = x
        screen_x = F.relu(self.screen_conv1(screen_x))
        screen_x = F.relu(self.screen_conv2(screen_x))
        minimap_x = F.relu(self.minimap_conv1(minimap_x))
        minimap_x = F.relu(self.minimap_conv2(minimap_x))
        x = torch.cat((screen_x, minimap_x), 1)
        s = F.relu(self.fc(x.view(x.size(0), -1)))
        value = self.value_fc(s)
        spatial_policy = self.spatial_policy_conv(x) * 3
        nonspatial_policy = self.nonspatial_policy_fc(s)
        policy = self._transform_policy(spatial_policy, nonspatial_policy)
        return policy, value

    def _transform_policy(self, sp, nonsp):
        spatial_policy = [chunk.contiguous().view(chunk.size(0), -1)
                          for chunk in sp.chunk(sp.size(1), dim=1)]
        idx, nonspatial_policy = 0, []
        for dim in self._out_dims:
            nonspatial_policy.append(nonsp[:, idx:idx+dim])
            idx += dim
        return nonspatial_policy[0:1] + spatial_policy + nonspatial_policy[1:]
