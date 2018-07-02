from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random
import math
from copy import deepcopy
import queue
import threading
import multiprocessing
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from gym.spaces import prng
from gym.spaces.discrete import Discrete

from sc2learner.agents.memory import ReplayMemory, Transition
from sc2learner.envs.spaces.mask_discrete import MaskDiscrete


def tuple_cuda(tensors):
  if isinstance(tensors, tuple):
    return tuple(tensor.pin_memory().cuda(async=True) for tensor in tensors)
  else:
    return tensors.pin_memory().cuda(async=True)


def tuple_variable(tensors, volatile=False):
  if isinstance(tensors, tuple):
    return tuple(Variable(tensor, volatile=volatile) for tensor in tensors)
  else:
    return Variable(tensors, volatile=volatile)


def actor_worker(pid, env_create_fn, q_network, difficulties, current_eps,
                 cur_difficulty_idx, action_space, allow_eval_mode,
                 transition_queue, outcome_queue, use_curriculum, discount):
  np.random.seed(pid)
  random.seed(pid)
  prng.seed(pid)

  def preprocess_observation(observation):
    action_mask = None
    if isinstance(action_space, MaskDiscrete):
      action_mask, observation = observation[-1], observation[:-1]
      if len(observation) == 1: observation = observation[0]
    return observation, action_mask

  def act(observation, eps=0):
    if random.uniform(0, 1) >= eps:
      if isinstance(observation, tuple):
        observation = tuple(torch.from_numpy(np.expand_dims(array, 0))
                            for array in observation)
      else:
        observation = torch.from_numpy(np.expand_dims(observation, 0))
      if torch.cuda.is_available(): observation = tuple_cuda(observation)
      if allow_eval_mode: q_network.eval()
      observation, action_mask = preprocess_observation(observation)
      q = q_network(tuple_variable(observation, volatile=True))
      if action_mask is not None: q[action_mask == 0] = float('-inf')
      action = q.data.max(1)[1][0]
      return action
    else:
      _, action_mask = preprocess_observation(observation)
      if action_mask is not None:
        return action_space.sample(np.nonzero(action_mask)[0])
      else:
        return action_space.sample()

  episode_id = 0
  while True:
    try:
      episode_id += 1
      cum_return = 0.0
      random_seed =  (pid * 11111111 + int(time.time() * 1000)) & 0xFFFFFFFF
      print("Random Seed: %d" % random_seed)
      if use_curriculum: cur_difficulty = difficulties[cur_difficulty_idx.value]
      else: cur_difficulty = random.choice(difficulties)
      env = env_create_fn(cur_difficulty, random_seed)
      observation = env.reset()
      done, n_frames, transition_list = False, 0, []
      while not done:
        action = act(observation, eps=current_eps.value)
        next_observation, reward, done, _ = env.step(action)
        transition_list.append(
            (observation, action, reward, next_observation, done))
        observation = next_observation
        cum_return += reward
        n_frames += 1

      discounted_return = 0
      for transition in reversed(transition_list):
        discounted_return = discounted_return * discount + transition[2]
        transition_queue.put(transition + (discounted_return,))
      outcome_queue.put(cum_return)
      env.close()
      print("Actor Worker ID: %d Episode: %d Frames %d Difficulty: %s "
            "Epsilon: %f Return: %f." % (pid, episode_id, n_frames,
                                         cur_difficulty, current_eps.value,
                                         cum_return))
      sys.stdout.flush()

    except:
      print("Expection occured in PID %d. Restart." % pid)


class DDQNAgent(object):
  '''Deep Q-learning agent.'''

  def __init__(self,
               observation_space,
               action_space,
               network,
               optimizer_type,
               learning_rate,
               momentum,
               adam_eps,
               batch_size,
               discount,
               eps_method,
               eps_start,
               eps_end,
               eps_decay,
               eps_decay2,
               memory_size,
               init_memory_size,
               frame_step_ratio,
               gradient_clipping,
               double_dqn,
               mmc_beta,
               mmc_discount,
               target_update_freq,
               winning_rate_threshold,
               difficulties,
               allow_eval_mode=True,
               loss_type='mse',
               init_model_path=None,
               save_model_dir=None,
               save_model_freq=50000,
               print_freq=1000):
    assert isinstance(action_space, Discrete)

    self._batch_size = batch_size
    self._discount = discount
    self._eps_method = eps_method
    self._eps_start = eps_start
    self._eps_end = eps_end
    self._eps_decay = eps_decay
    self._eps_decay2 = eps_decay2
    self._frame_step_ratio = frame_step_ratio
    self._target_update_freq = target_update_freq
    self._double_dqn = double_dqn
    self._save_model_dir = save_model_dir
    self._save_model_freq = save_model_freq
    self._action_space = action_space
    self._memory_size = memory_size
    self._init_memory_size = max(init_memory_size, batch_size)
    self._gradient_clipping = gradient_clipping
    self._allow_eval_mode = allow_eval_mode
    self._loss_type = loss_type
    self._print_freq = print_freq
    self._mmc_beta = mmc_beta
    self._mmc_discount = mmc_discount
    self._episode_idx = 0
    self._current_eps = multiprocessing.Value('d', eps_start)
    self._num_threads = 4

    self._winning_rate_threshold = winning_rate_threshold
    self._difficulties = difficulties
    self._cur_difficulty_idx = multiprocessing.Value('i', 0)
    self._recent_outcomes = deque(maxlen=1000)

    self._q_network = network
    self._q_network.share_memory()
    self._target_q_network = deepcopy(network)
    if init_model_path:
      self._load_model(init_model_path)
      self._episode_idx = int(init_model_path[init_model_path.rfind('-')+1:])
    if torch.cuda.device_count() > 1:
      self._q_network = nn.DataParallel(self._q_network)
      self._target_q_network = nn.DataParallel(self._target_q_network)
    if torch.cuda.is_available():
      self._q_network.cuda()
      self._target_q_network.cuda()
    if self._allow_eval_mode:
      self._target_q_network.eval()

    self._update_target_network()

    if optimizer_type == "rmsprop":
      self._optimizer = optim.RMSprop(self._q_network.parameters(),
                                      momentum=momentum,
                                      lr=learning_rate)
    elif optimizer_type == "adam":
      self._optimizer = optim.Adam(self._q_network.parameters(),
                                   eps=adam_eps,
                                   lr=learning_rate)
    elif optimizer_type == "sgd":
      self._optimizer = optim.SGD(self._q_network.parameters(),
                                  momentum=momentum,
                                  lr=learning_rate)
    else:
      raise NotImplementedError

  def act(self, observation, eps=0):
    if random.uniform(0, 1) >= eps:
      if isinstance(observation, tuple):
        observation = tuple(torch.from_numpy(np.expand_dims(array, 0))
                            for array in observation)
      else:
        observation = torch.from_numpy(np.expand_dims(observation, 0))
      if torch.cuda.is_available():
        observation = tuple_cuda(observation)
      if self._allow_eval_mode:
        self._q_network.eval()

      observation, action_mask = self._preprocess_observation(observation)
      q = self._q_network(tuple_variable(observation, volatile=True))
      if action_mask is not None: q[action_mask == 0] = float('-inf')
      action = q.data.max(1)[1][0]
      return action
    else:
      _, action_mask = self._preprocess_observation(observation)
      if action_mask is not None:
        return self._action_space.sample(np.nonzero(action_mask)[0])
      else:
        return self._action_space.sample()

  def learn(self, create_env_fn, num_actor_workers, use_curriculum):
    self._init_parallel_actors(create_env_fn, num_actor_workers, use_curriculum)
    steps, loss_sum = 0, 0.0
    t = time.time()
    while True:
      if steps % self._target_update_freq == 0: self._update_target_network()
      self._current_eps.value = self._get_current_eps(steps)
      self._update_curriculum_difficulty()
      loss_sum += self._optimize()
      steps += 1
      if self._save_model_dir and steps % self._save_model_freq == 0:
        self._save_model(
            os.path.join(self._save_model_dir, 'agent.model-%d' % steps))
      if steps % self._print_freq == 0:
        print("Steps: %d Time: %f Epsilon: %f Loss %f " % (
            steps, time.time() - t, self._current_eps.value,
            loss_sum / self._print_freq))
        loss_sum = 0.0
        t = time.time()

  def _preprocess_observation(self, observation):
    action_mask = None
    if isinstance(self._action_space, MaskDiscrete):
      action_mask = observation[-1]
      observation = observation[:-1]
      if len(observation) == 1: observation = observation[0]
    return observation, action_mask

  def _optimize(self):
    #print("Batch Queue Size: %d" % self._batch_queue.qsize())
    (next_obs_batch, obs_batch, reward_batch, action_batch, done_batch, \
        return_batch) = self._batch_queue.get()
    next_obs_batch, action_mask = self._preprocess_observation(next_obs_batch)
    obs_batch, _ = self._preprocess_observation(obs_batch)
    # compute max-q target
    if self._allow_eval_mode: self._q_network.eval()
    q_next_target = self._target_q_network(next_obs_batch)
    if self._double_dqn:
      q_next = self._q_network(next_obs_batch)
      if action_mask is not None: q_next[action_mask == 0] = float('-inf')
      futures = q_next_target.gather(
          1, q_next.max(dim=1)[1].view(-1, 1)).squeeze()
    else:
      if action_mask is not None:
        q_next_target[action_mask == 0] = float('-inf')
      futures = q_next_target.max(dim=1)[0].view(-1, 1).squeeze()
    futures = futures * (1 - done_batch)
    target_q = reward_batch + self._discount * futures
    target_q = target_q * self._mmc_beta + (1.0 - self._mmc_beta) * return_batch
    # define loss
    self._q_network.train()
    q = self._q_network(obs_batch).gather(1, action_batch.view(-1, 1))
    if self._loss_type == "smooth_l1":
      loss = F.smooth_l1_loss(q, Variable(target_q.data))
    elif self._loss_type == "mse":
      loss = F.mse_loss(q, Variable(target_q.data))
    else:
      raise NotImplementedError
    # compute gradient and update parameters
    self._optimizer.zero_grad()
    loss.backward()
    for param in self._q_network.parameters():
      param.grad.data.clamp_(-self._gradient_clipping, self._gradient_clipping)
    self._optimizer.step()
    return loss.data[0]

  def _init_parallel_actors(self, create_env_fn, num_actor_workers,
                            use_curriculum):
    multiprocessing.set_start_method('spawn')
    self._transition_queue = multiprocessing.Queue(8)
    self._outcome_queue = multiprocessing.Queue(200000)
    self._actor_processes = [
        multiprocessing.Process(target=actor_worker,
                                args=(pid,
                                      create_env_fn,
                                      self._q_network,
                                      self._difficulties,
                                      self._current_eps,
                                      self._cur_difficulty_idx,
                                      self._action_space,
                                      self._allow_eval_mode,
                                      self._transition_queue,
                                      self._outcome_queue,
                                      use_curriculum,
                                      self._mmc_discount))
        for pid in range(num_actor_workers)
    ]
    self._batch_queue = queue.Queue(8)
    self._batch_thread = [threading.Thread(target=self._prepare_batch,
                                           args=(tid,))
                          for tid in range(self._num_threads)]
    for process in self._actor_processes:
      process.daemon = True
      process.start()
      time.sleep(3.0)
    for thread in self._batch_thread:
      thread.daemon = True
      thread.start()
      time.sleep(0.2)

  def _update_curriculum_difficulty(self):
    has_new_outcome = False
    while not self._outcome_queue.empty():
      self._recent_outcomes.append(self._outcome_queue.get())
      has_new_outcome = True
    if (has_new_outcome and
        len(self._recent_outcomes) == self._recent_outcomes.maxlen):
      winning_rate = (float(sum(self._recent_outcomes)) / \
          len(self._recent_outcomes) + 1.0) / 2.0
      print("Difficulty: %s Winning_rate %f:" % (
          self._difficulties[self._cur_difficulty_idx.value], winning_rate))
      if (winning_rate >= self._winning_rate_threshold and
          len(self._difficulties) > self._cur_difficulty_idx.value + 1):
        self._cur_difficulty_idx.value += 1
        self._recent_outcomes.clear()

  def _prepare_batch(self, tid):
    memory = ReplayMemory(int(self._memory_size / self._num_threads))
    steps = 0
    if self._frame_step_ratio < 1:
      steps_per_frame = int(1 / self._frame_step_ratio)
    else:
      frames_per_step = int(self._frame_step_ratio)
    while True:
      steps += 1
      if self._frame_step_ratio < 1:
        if steps % steps_per_frame == 0: 
          memory.push(*(self._transition_queue.get()))
      else:
        #print("Trans Queue Size: %d" % self._transition_queue.qsize())
        for i in range(frames_per_step):
          memory.push(*(self._transition_queue.get()))
      if len(memory) < self._init_memory_size / self._num_threads: continue
      transitions = memory.sample(self._batch_size)
      self._batch_queue.put(self._transitions_to_batch(transitions))

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
    return_batch = torch.FloatTensor(batch.mc_return)

    # move to cuda
    if torch.cuda.is_available():
      next_obs_batch = tuple_cuda(next_obs_batch)
      obs_batch = tuple_cuda(obs_batch)
      reward_batch = tuple_cuda(reward_batch)
      action_batch = tuple_cuda(action_batch)
      done_batch = tuple_cuda(done_batch)
      return_batch = tuple_cuda(return_batch)

    # create variables
    next_obs_batch = tuple_variable(next_obs_batch, volatile=True)
    obs_batch = tuple_variable(obs_batch)
    reward_batch = tuple_variable(reward_batch)
    action_batch = tuple_variable(action_batch)
    done_batch = tuple_variable(done_batch)
    return_batch = tuple_variable(return_batch)
    print(obs_batch.size(), next_obs_batch.size(), reward_batch.size(), action_batch.size(), done_batch.size(), return_batch.size())

    return (next_obs_batch, obs_batch, reward_batch, action_batch, done_batch,
            return_batch)

  def _update_target_network(self):
    self._target_q_network.load_state_dict(self._q_network.state_dict())
        
  def _save_model(self, model_path):
    torch.save(self._q_network.state_dict(), model_path)

  def _load_model(self, model_path):
    self._q_network.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage))

  def _get_current_eps(self, steps):
    if self._eps_method == 'exponential':
      eps = self._eps_end + (self._eps_start - self._eps_end) * \
          math.exp(-1. * steps / self._eps_decay)
    elif self._eps_method == 'linear':
      if steps < self._eps_decay:
        eps = self._eps_start - (self._eps_start - self._eps_end) * \
            steps / self._eps_decay
      elif steps < self._eps_decay2:
        eps = self._eps_end - (self._eps_end - 0.01) * \
            (steps - self._eps_decay) / self._eps_decay2
      else:
        eps = 0.01
    else:
      raise NotImplementedError
    return eps
