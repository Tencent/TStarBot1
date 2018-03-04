import os
import time
import random
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from agents.memory import ReplayMemory, Transition


def tuple_cuda(tensors):
    if isinstance(tensors, tuple):
        return tuple(tensor.cuda() for tensor in tensors)
    else:
        return tensors.cuda()


def tuple_variable(tensors, volatile=False):
    if isinstance(tensors, tuple):
        return tuple(Variable(tensor, volatile=volatile)
                for tensor in tensors)
    else:
        return Variable(tensors, volatile=volatile)


class DQNAgent(object):
    '''Deep Q-learning agent.'''

    def __init__(self,
                 observation_space,
                 action_space,
                 network,
                 learning_rate,
                 batch_size,
                 discount,
                 eps_start,
                 eps_end,
                 eps_decay,
                 memory_size,
                 init_model_path=None,
                 save_model_dir=None,
                 save_model_freq=1000):
        self._batch_size = batch_size
        self._discount = discount
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._save_model_dir = save_model_dir
        self._save_model_freq = save_model_freq
        self._action_space = action_space
        self._episode_idx = 0

        self._q_network = network
        if init_model_path:
            self._load_model(init_model_path)
            self._episode_idx = int(init_model_path[
                init_model_path.rfind('-')+1:])
        if torch.cuda.device_count() > 1:
            self._q_network = nn.DataParallel(self._q_network)
        if torch.cuda.is_available():
            self._q_network.cuda()

        self._optimizer = optim.RMSprop(self._q_network.parameters(),
                                        lr=learning_rate)
        self._memory = ReplayMemory(memory_size)

    def act(self, observation, eps=0):
        if random.uniform(0, 1) >= eps:
            if isinstance(observation, tuple):
                observation = tuple(torch.from_numpy(np.expand_dims(array, 0))
                                    for array in observation)
            else:
                observation = torch.from_numpy(np.expand_dims(observation, 0))
            if torch.cuda.is_available():
                observation = tuple_cuda(observation)
            q = self._q_network(tuple_variable(observation, volatile=True))
            action = q.data.max(1)[1][0]
            return action
        else:
            return self._action_space.sample()

    def learn(self, env):
        t = time.time()
        steps = 0
        while True:
            self._episode_idx += 1
            loss_sum, loss_count, cum_return = 0.0, 1e-20, 0.0
            observation = env.reset()
            done = False
            while not done:
                action = self.act(observation, eps=self._get_current_eps(steps))
                next_observation, reward, done, _ = env.step(action)
                self._memory.push(observation, action, reward,
                                  next_observation, done)
                if len(self._memory) > self._batch_size:
                    loss_sum += self._optimize()
                    loss_count += 1
                observation = next_observation
                cum_return += reward
                steps += 1

            if (self._save_model_dir and
                self._episode_idx % self._save_model_freq == 0):
                self._save_model(os.path.join(
                    self._save_model_dir, 'agent.model-%d' % self._episode_idx))
            print("Episode %d Steps: %d Time: %f Eps: %f Loss %f Return: %f." %
                  (self._episode_idx,
                   steps,
                   time.time() - t,
                   self._get_current_eps(steps),
                   loss_sum / loss_count,
                   cum_return))
            t = time.time()

    def _optimize(self):
        assert len(self._memory) >= self._batch_size
        transitions = self._memory.sample(self._batch_size)
        (next_obs_batch, obs_batch, reward_batch, action_batch, done_batch) = \
            self._transitions_to_batch(transitions)
        # compute max-q target
        q_next = self._q_network(next_obs_batch)
        futures = q_next.max(dim=1)[0].view(-1, 1).squeeze()
        futures = futures * (1 - done_batch)
        target_q = reward_batch + self._discount * futures
        target_q.volatile = False
        # compute gradient
        q = self._q_network(obs_batch).gather(
            1, action_batch.view(-1, 1))
        loss = F.smooth_l1_loss(q, target_q)
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        # update parameters
        self._optimizer.step()
        return loss.data[0]

    def _transitions_to_batch(self, transitions):
        # batch to pytorch tensor
        batch = Transition(*zip(*transitions))
        if isinstance(batch.next_observation[0], tuple):
            next_obs_batch = tuple(torch.from_numpy(np.stack(feat))
                                   for feat in zip(*batch.next_observation))
        else:
            next_obs_batch = torch.from_numpy(np.stack(batch.next_observation))
        if isinstance(batch.observation[0], tuple):
            obs_batch = tuple(torch.from_numpy(np.stack(feat))
                              for feat in zip(*batch.observation))
        else:
            obs_batch = torch.from_numpy(np.stack(batch.observation))
        reward_batch = torch.FloatTensor(batch.reward)
        action_batch = torch.LongTensor(batch.action)
        done_batch = torch.Tensor(batch.done)

        # move to cuda
        if torch.cuda.is_available():
            next_obs_batch = tuple_cuda(next_obs_batch)
            obs_batch = tuple_cuda(obs_batch)
            reward_batch = tuple_cuda(reward_batch)
            action_batch = tuple_cuda(action_batch)
            done_batch = tuple_cuda(done_batch)

        # create variables
        next_obs_batch = tuple_variable(next_obs_batch, volatile=True)
        obs_batch = tuple_variable(obs_batch)
        reward_batch = tuple_variable(reward_batch)
        action_batch = tuple_variable(action_batch)
        done_batch = tuple_variable(done_batch)

        return (next_obs_batch, obs_batch, reward_batch, action_batch,
                done_batch)
                
    def _save_model(self, model_path):
        torch.save(self._q_network.state_dict(), model_path)

    def _load_model(self, model_path):
        self._q_network.load_state_dict(
            torch.load(model_path, map_location=lambda storage, loc: storage))

    def _get_current_eps(self, steps):
        eps = self._eps_end + (self._eps_start - self._eps_end) * \
            math.exp(-1. * steps / self._eps_decay)
        return eps
