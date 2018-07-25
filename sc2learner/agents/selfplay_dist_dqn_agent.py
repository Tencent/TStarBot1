from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import struct
import random
import math
from copy import deepcopy
import queue
from threading import Thread
from collections import deque
import io
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from gym.spaces import prng
from gym.spaces.discrete import Discrete
from gym import spaces
from memoire import ReplayMemoryClient
from memoire import ReplayMemoryServer
from memoire import Bind
from memoire import Conn

from sc2learner.envs.actions.zerg_action_agent_wrappers import ZergActionAgentWrapper
from sc2learner.envs.observations.zerg_observation_agent_wrappers import ZergNonspatialObservationAgentWrapper
from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.utils.utils import tprint


class Actor(object):

  def __init__(self, network, action_space):
    self._action_space = action_space
    self._network = network
    if torch.cuda.device_count() > 1:
      self._network = nn.DataParallel(self._network)
    if torch.cuda.is_available(): self._network.cuda()
    self._optimizer = None
    self._target_network = None
    self._num_optim_steps = 0
    self._is_network_loaded = False
    self._cur_epsilon = 1.0

  def act(self, observation, eps=None):
    if eps is None: eps = self._cur_epsilon
    self._network.eval()
    if random.uniform(0, 1) >= eps:
      observation = torch.from_numpy(np.expand_dims(observation, 0))
      if torch.cuda.is_available():
        observation = observation.pin_memory().cuda(non_blocking=True)
      with torch.no_grad():
        q = self._network(observation)
        action = q.data.max(1)[1].item()
      return action
    else:
      return self._action_space.sample()

  def reset(self, observation):
    pass

  def optimize_step(self,
                    obs_batch,
                    next_obs_batch,
                    action_batch,
                    reward_batch,
                    done_batch,
                    mc_return_batch,
                    discount,
                    mmc_beta,
                    gradient_clipping,
                    adam_eps,
                    learning_rate,
                    target_update_freq):
    # create optimizer
    if self._optimizer is None:
      self._optimizer = optim.Adam(self._network.parameters(),
                                   eps=adam_eps,
                                   lr=learning_rate)
    # create target network
    if self._target_network is None:
      self._target_network = deepcopy(self._network)
      if torch.cuda.is_available(): self._target_network.cuda()
      self._target_network.eval()

    # update target network
    if self._num_optim_steps % target_update_freq == 0:
      self._target_network.load_state_dict(self._network.state_dict())

    # move to gpu
    if torch.cuda.is_available():
      obs_batch = obs_batch.cuda(non_blocking=True)
      next_obs_batch = next_obs_batch.cuda(non_blocking=True)
      action_batch = action_batch.cuda(non_blocking=True)
      reward_batch = reward_batch.cuda(non_blocking=True)
      mc_return_batch = mc_return_batch.cuda(non_blocking=True)
      done_batch = done_batch.cuda(non_blocking=True)

    # compute max-q target
    self._network.eval()
    with torch.no_grad():
      q_next_target = self._target_network(next_obs_batch)
      q_next = self._network(next_obs_batch)
      futures = q_next_target.gather(
          1, q_next.max(dim=1)[1].view(-1, 1)).squeeze()
      futures = futures * (1 - done_batch)
      target_q = reward_batch + discount * futures
      target_q = target_q * mmc_beta + (1.0 - mmc_beta) * mc_return_batch

    # define loss
    self._network.train()
    q = self._network(obs_batch).gather(1, action_batch.view(-1, 1)).squeeze()
    loss = F.mse_loss(q, target_q.detach())

    # compute gradient and update parameters
    self._optimizer.zero_grad()
    loss.backward()
    for param in self._network.parameters():
      param.grad.data.clamp_(-gradient_clipping, gradient_clipping)
    self._optimizer.step()
    self._num_optim_steps += 1
    return loss.data.item()

  def load_network(self, state_dict):
    self._network.load_state_dict(state_dict)
    self._is_network_loaded = True

  def load_epsilon(self, epsilon):
    self._cur_epsilon = epsilon

  @property
  def is_network_loaded(self):
    return self._is_network_loaded

  @property
  def network(self):
    return self._network

  def _tuple_tensor_from_numpy(self, tensors):
    if isinstance(tensors, tuple):
      return tuple(torch.from_numpy(np.expand_dims(tensor, 0))
                   for tensor in tensors)
    else:
      return torch.from_numpy(np.expand_dims(tensors, 0))

  def _tuple_cuda(self, tensors):
    if isinstance(tensors, tuple):
      return tuple(tensor.pin_memory().cuda(async=True) for tensor in tensors)
    else:
      return tensors.pin_memory().cuda(async=True)

  def _parse_observation(self, observation):
    action_mask = None
    if isinstance(self._action_space, MaskDiscrete):
      action_mask, observation = observation[-1], observation[:-1]
      if len(observation) == 1: observation = observation[0]
    return observation, action_mask


class DistRolloutWorker(object):

  def __init__(self,
               memory_size,
               difficulties,
               env_create_fn,
               network,
               action_space,
               push_freq,
               learner_ip="localhost"):
    self._actor = Actor(network, action_space)
    self._cur_epsilon = 1.0
    self._difficulties = difficulties
    self._env_create_fn = env_create_fn
    self._push_freq = push_freq
    self._num_steps = 0
    self._num_rollouts = 0
    self._memory_client, self._threads = self._setup_memory_client(
        learner_ip, memory_size)
    self._replay_memory = self._memory_client.rem
    self._cache_size = self._replay_memory.cache_size
    self._set_random_seed(self._replay_memory.uuid)

  def run(self):
    while not self._actor.is_network_loaded:
      continue
      time.sleep(1)

    while True:
      try:
        self._rollout()
      except KeyboardInterrupt:
        break
      except Exception as e:
        tprint("[Rollout Exception]: %s" % e)
        continue

  def _set_random_seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    prng.seed(seed)

  def _setup_memory_client(self,
                           server_ip,
                           memory_size):
    client = ReplayMemoryClient("tcp://%s:5960" % server_ip,
                                "tcp://%s:5961" % server_ip,
                                "tcp://%s:5962" % server_ip,
                                memory_size)
    client.rem.print_info()

    def _update_network_worker(client):
      while True:
        try:
          bytes_recv = client.sub_bytes('model')
          self._cur_epsilon = struct.unpack('d', bytes_recv[:8])[0]
          f = io.BytesIO(bytes_recv[8:])
          self._actor.load_network(
              torch.load(f, map_location=lambda storage, loc: storage))
          tprint("Network updated. Epsilon = %f" % self._cur_epsilon)
        except Exception as e:
          tprint("[Update Network Exception]: %s" % e)
          continue

    #def _update_network_worker(client):
      #while True:
        #try:
          #f = io.BytesIO(client.sub_bytes('model'))
          #self._actor.load_network(
              #torch.load(f, map_location=lambda storage, loc: storage))
          #tprint("Network updated.")
          #self._cur_epsilon = float(client.sub_bytes('epsilon'))
          #tprint("Epsilon updated = %f."  % self._cur_epsilon)
        #except Exception as e:
          #tprint("[Update Network Exception]: %s" % e)
          #continue

    threads = [
        Thread(target=_update_network_worker, args=(client,))
    ]
    for thread in threads:
      thread.start()
    return client, threads

  def _rollout(self):
    entry_reward = np.ndarray((1), dtype=np.float32)
    entry_action = np.ndarray((1), dtype=np.float32)
    entry_reward = np.ndarray((1), dtype=np.float32)
    entry_prob = np.ndarray((1), dtype=np.float32)
    entry_value = np.ndarray((1), dtype=np.float32)

    random_seed =  random.randint(0, 2**32 - 1)
    difficulty = random.choice(self._difficulties)
    env = self._env_create_fn(difficulty, random_seed)
    self._replay_memory.new_episode()
    observation = env.reset()
    done = False
    while not done:
      entry_state = observation
      action = self._actor.act(observation, eps=self._cur_epsilon)
      observation, reward, done, info = env.step(action)
      entry_action[0], entry_reward[0]= action, reward
      entry_prob[0] = False # entry_prob is reused as terminal
      entry_value[0] = reward # entry_value is reused as reward
      self._replay_memory.add_entry(entry_state, entry_action, entry_reward,
                                    entry_prob, entry_value, weight=1.0)
      self._num_steps += 1
      if (self._num_rollouts > 0 and
          self._num_steps % int(self._cache_size / self._push_freq) == 0):
        self._memory_client.push_cache()
    env.close()
    entry_action[0], entry_reward[0], entry_value[0] = -1, 0, 0
    entry_prob[0] = True
    entry_state = observation
    self._replay_memory.add_entry(entry_state, entry_action, entry_reward,
                                  entry_prob, entry_value, weight=1.0)
    self._replay_memory.close_episode()
    self._memory_client.update_counter()
    tprint("Actor uuid: %d Seed: %d Difficulty: %s Epsilon: %f Outcome: %f." %
        (self._replay_memory.uuid, random_seed, difficulty, self._cur_epsilon,
         reward))
    self._num_rollouts += 1


class SelfplayDistRolloutWorker(object):

  def __init__(self,
               env,
               network,
               memory_size,
               push_freq,
               model_cache_prob,
               model_cache_size,
               learner_ip="localhost",
               game_version="3.16.1"):
    self._env = env
    self._actor = Actor(network, self._env.action_space)
    self._raw_oppo_actor = Actor(deepcopy(network), self._env.action_space)
    self._oppo_actor = ZergNonspatialObservationAgentWrapper(
        self._raw_oppo_actor, self._env.action_space)
    self._oppo_actor = ZergActionAgentWrapper(self._oppo_actor, game_version)
    self._env.register_opponent(self._oppo_actor)

    self._model_cache = deque(maxlen=model_cache_size)
    self._model_cache_prob = model_cache_prob
    self._latest_model = None

    self._cur_epsilon = 1.0
    self._push_freq = push_freq
    self._num_steps = 0
    self._num_rollouts = 0
    self._memory_client, self._threads = self._setup_memory_client(
        learner_ip, memory_size)
    self._replay_memory = self._memory_client.rem
    self._cache_size = self._replay_memory.cache_size
    self._set_random_seed(self._replay_memory.uuid)

  def run(self):
    while not self._actor.is_network_loaded:
      continue
      time.sleep(1)

    while True:
      try:
        self._update_opponent()
        self._rollout()
      except KeyboardInterrupt:
        break
      except Exception as e:
        tprint("[Rollout Exception]: %s" % e)
        continue

  def _update_opponent(self):
    if random.uniform(0, 1.0) < 0.8 or len(self._model_cache) == 0:
      self._raw_oppo_actor.load_network(self._latest_model)
      self._raw_oppo_actor.load_epsilon(self._cur_epsilon)
    else:
      model_dict, epsilon = random.choice(self._model_cache)
      self._raw_oppo_actor.load_network(model_dict)
      self._raw_oppo_actor.load_epsilon(epsilon)

  def _set_random_seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    prng.seed(seed)

  def _setup_memory_client(self,
                           server_ip,
                           memory_size):
    client = ReplayMemoryClient("tcp://%s:5960" % server_ip,
                                "tcp://%s:5961" % server_ip,
                                "tcp://%s:5962" % server_ip,
                                memory_size)
    client.rem.print_info()

    def _update_network_worker(client):
      while True:
        try:
          bytes_recv = client.sub_bytes('model')
          self._cur_epsilon = struct.unpack('d', bytes_recv[:8])[0]
          f = io.BytesIO(bytes_recv[8:])
          model_dict = torch.load(f, map_location=lambda storage, loc: storage)
          self._cache_model(model_dict, self._cur_epsilon)
          self._actor.load_network(model_dict)
          tprint("Network updated. Epsilon = %f" % self._cur_epsilon)
        except Exception as e:
          tprint("[Update Network Exception]: %s" % e)
          continue

    threads = [
        Thread(target=_update_network_worker, args=(client,))
    ]
    for thread in threads:
      thread.start()
    return client, threads

  def _cache_model(self, model_dict, epsilon):
    self._latest_model = model_dict
    if random.uniform(0, 1.0) < self._model_cache_prob:
      self._model_cache.append((model_dict, epsilon))
      tprint("New model cached. Number of current cached models: %d" %
          len(self._model_cache))

  def _rollout(self):
    entry_reward = np.ndarray((1), dtype=np.float32)
    entry_action = np.ndarray((1), dtype=np.float32)
    entry_reward = np.ndarray((1), dtype=np.float32)
    entry_prob = np.ndarray((1), dtype=np.float32)
    entry_value = np.ndarray((1), dtype=np.float32)

    self._replay_memory.new_episode()
    observation = self._env.reset()
    done = False
    while not done:
      entry_state = observation
      action = self._actor.act(observation, eps=self._cur_epsilon)
      observation, reward, done, info = self._env.step(action)
      entry_action[0], entry_reward[0]= action, reward
      entry_prob[0] = False # entry_prob is reused as terminal
      entry_value[0] = reward # entry_value is reused as reward
      self._replay_memory.add_entry(entry_state, entry_action, entry_reward,
                                    entry_prob, entry_value, weight=1.0)
      self._num_steps += 1
      if (self._num_rollouts > 0 and
          self._num_steps % int(self._cache_size / self._push_freq) == 0):
        self._memory_client.push_cache()
    entry_action[0], entry_reward[0], entry_value[0] = -1, 0, 0
    entry_prob[0] = True
    entry_state = observation
    self._replay_memory.add_entry(entry_state, entry_action, entry_reward,
                                  entry_prob, entry_value, weight=1.0)
    self._replay_memory.close_episode()
    self._memory_client.update_counter()
    tprint("Actor uuid: %d Epsilon: %f Outcome: %f." %
        (self._replay_memory.uuid, self._cur_epsilon, reward))
    self._num_rollouts += 1


class DistDDQNLearner(object):

  def __init__(self,
               network,
               observation_space,
               action_space,
               num_caches,
               cache_size,
               num_pull_workers,
               discount,
               eps_start,
               eps_end,
               eps_decay,
               eps_decay2,
               init_checkpoint_path="",
               priority_exponent=0.0):
    state_size = observation_space.spaces[0].shape[0] \
        if isinstance(observation_space, spaces.Tuple) \
        else observation_space.shape[0]
    self._memory_server, self._threads = self._setup_memory_server(
        state_size=state_size,
        cache_size=cache_size,
        num_caches=num_caches,
        num_pull_workers=num_pull_workers,
        priority_exponent=priority_exponent,
        discount=discount)
    self._discount = discount
    self._eps_start = eps_start
    self._eps_end = eps_end
    self._eps_decay = eps_decay
    self._eps_decay2 = eps_decay2
    self._cur_epsilon = eps_start

    self._actor = Actor(network, action_space)
    if init_checkpoint_path:
      self._load_model(init_checkpoint_path)

    self._publish_model()
    time.sleep(5)
    self._publish_model()
    time.sleep(5)

  def learn(self,
            batch_size,
            mmc_beta,
            gradient_clipping,
            adam_eps,
            learning_rate,
            warmup_size,
            target_update_freq,
            checkpoint_dir,
            checkpoint_freq,
            print_freq):
    batch_queue = queue.Queue(8)
    batch_threads = [
        Thread(target=self._batch_worker, args=(batch_queue, batch_size,
                                                warmup_size)),
        Thread(target=self._publish_model_worker)
    ]
    for thread in batch_threads:
      thread.start()

    num_updates, loss_sum = 1, 0.0
    t = time.time()
    while True:
      observation, next_observation, action, reward, done, mc_return = \
          batch_queue.get()
      self._cur_epsilon = self._schedule_epsilon(num_updates)
      loss_sum += self._actor.optimize_step(
          obs_batch=observation,
          next_obs_batch=next_observation,
          action_batch=action,
          reward_batch=reward,
          done_batch=done,
          mc_return_batch=mc_return,
          discount=self._discount,
          mmc_beta=mmc_beta,
          gradient_clipping=gradient_clipping,
          adam_eps=adam_eps,
          learning_rate=learning_rate,
          target_update_freq=target_update_freq)
      if num_updates % checkpoint_freq == 0:
        ckpt_path = os.path.join(checkpoint_dir, 'agent.model-%d' % num_updates)
        self._save_checkpoint(ckpt_path)
      if num_updates % print_freq == 0:
        tprint("Steps: %d Time: %f Loss %f Actor Steps: %d Current Eps: %f" % (
            num_updates, time.time() - t, loss_sum / print_freq,
            self._memory_server.total_steps, self._cur_epsilon))
        loss_sum = 0.0
        t = time.time()
      num_updates += 1

  def _publish_model_worker(self):
    while True:
      self._publish_model()
      time.sleep(10)

  def _batch_worker(self, batch_queue, batch_size, warmup_size):
    while True:
      try:
        if self._memory_server.total_steps <= warmup_size:
          time.sleep(5)
          tprint("Warming up: %d frames." % self._memory_server.total_steps)
          continue
        prev_trans, next_trans, weight = self._memory_server.get_batch(batch_size)
        t = time.time()
        observation = prev_trans[0].squeeze(1)
        next_observation = next_trans[0].squeeze(1)
        action = prev_trans[1].squeeze()
        mc_return = prev_trans[2].squeeze()
        reward = prev_trans[4].squeeze()
        done = next_trans[3].squeeze()

        if np.any(action < 0):
          np.set_printoptions(threshold=np.nan, linewidth=300)
          tprint("Error action detected: actions %s, weights: %s" %
              (action, weight))
          continue

        observation = torch.from_numpy(observation)
        next_observation = torch.from_numpy(next_observation)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        mc_return = torch.FloatTensor(mc_return)
        done = torch.FloatTensor(done)

        if torch.cuda.is_available():
          observation = observation.pin_memory()
          next_observation = next_observation.pin_memory()
          action = action.pin_memory()
          reward = reward.pin_memory()
          mc_return = mc_return.pin_memory()
          done = done.pin_memory()

        batch_queue.put(
            (observation, next_observation, action, reward, done, mc_return))
      except RuntimeError:
        time.sleep(0.001)
        continue

  def _setup_memory_server(self,
                           state_size,
                           cache_size,
                           num_caches,
                           num_pull_workers,
                           priority_exponent,
                           discount):
    server = ReplayMemoryServer(state_size=state_size,
                                action_size=1,
                                reward_size=1,
                                prob_size=1,# reused as done
                                value_size=1,
                                max_step=0,
                                n_caches=num_caches,
                                pub_endpoint="tcp://*:5960")
    server.rem.priority_exponent = priority_exponent
    server.rem.mix_lambda = 1 # used as MC return
    server.rem.frame_stack = 1
    server.rem.multi_step = 1
    server.rem.cache_size = cache_size
    server.rem.discount_factor = [discount]
    server.rem.reward_coeff = [1.0]
    server.rem.cache_flags = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    server.print_info()

    threads = [
        Thread(target=server.rep_worker_main, args=("tcp://*:5961", Bind)),
        Thread(target=server.pull_proxy_main,
               args=("tcp://*:5962", "inproc://pull_workers"))
    ] + [
        Thread(target=server.pull_worker_main,
               args=("inproc://pull_workers", Conn))
        for _ in range(num_pull_workers)
    ]
    for thread in threads:
      thread.start()
    return server, threads

  def _publish_model(self):
    f = io.BytesIO()
    if torch.cuda.device_count() > 1:
      torch.save(self._actor.network.module.state_dict(), f)
    else:
      torch.save(self._actor.network.state_dict(), f)
    self._memory_server.pub_bytes(
      'model', struct.pack('d', self._cur_epsilon) + f.getvalue())
    #self._memory_server.pub_bytes('epsilon', str(self._cur_epsilon))

  def _save_checkpoint(self, checkpoint_path):
    if torch.cuda.device_count() > 1:
      torch.save(self._actor.network.module.state_dict(), checkpoint_path)
    else:
      torch.save(self._actor.network.state_dict(), checkpoint_path)

  def _load_model(self, model_path):
    self._actor.load_network(
        torch.load(model_path, map_location=lambda storage, loc: storage))

  def _schedule_epsilon(self, steps):
    if steps < self._eps_decay:
      return self._eps_start - (self._eps_start - self._eps_end) * \
          steps / self._eps_decay
    elif steps < self._eps_decay2:
      return self._eps_end - (self._eps_end - 0.01) * \
          (steps - self._eps_decay) / self._eps_decay2
    else:
      return 0.01
