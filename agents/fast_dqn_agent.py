import os
import time
from copy import deepcopy
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
#import torch.multiprocessing as multiprocessing
import multiprocessing
from gym import wrappers
from gym import spaces
from agents.memory import ReplayMemory, Transition


class FastDQNAgent(object):
    '''Deep Q-learning agent.'''

    def __init__(self,
                 observation_spec,
                 action_spec,
                 rmsprop_lr=1e-4,
                 rmsprop_eps=1e-5,
                 batch_size=128,
                 discount=1.0,
                 epsilon_max=1.0,
                 epsilon_min=0.1,
                 epsilon_decrease_steps=1000000,
                 memory_size=1000000,
                 target_update_freq=10000,
                 use_tiny_net=False,
                 use_gpu=True,
                 init_model_path=None,
                 save_model_dir=None,
                 save_model_freq=10000,
                 print_freq=1000,
                 enable_batchnorm=False,
                 seed=0):
        self._batch_size = batch_size
        self._discount = discount
        self._epsilon = epsilon_max
        self._epsilon_min = epsilon_min
        self._epsilon_decrease = (epsilon_max - epsilon_min) \
            / epsilon_decrease_steps
        self._action_spec = action_spec
        self._use_gpu = use_gpu
        self._save_model_dir = save_model_dir
        self._save_model_freq = save_model_freq
        self._print_freq = print_freq
        self._target_update_freq = target_update_freq
        self._steps = 0
        
        torch.manual_seed(seed)
        if use_gpu: torch.cuda.manual_seed(seed)

        if use_tiny_net:
            self._q_network = FullyConvNetTiny(
                resolution=observation_spec[2],
                in_channels_screen=observation_spec[0],
                in_channels_minimap=observation_spec[1],
                out_dims=action_spec[0],
                enable_batchnorm=enable_batchnorm)
        else:
            self._q_network = FullyConvNet(
                resolution=observation_spec[2],
                in_channels_screen=observation_spec[0],
                in_channels_minimap=observation_spec[1],
                out_dims=action_spec[0],
                enable_batchnorm=enable_batchnorm)
        self._q_network.apply(weights_init)
        self._q_network.share_memory()
        if init_model_path:
            self._load_model(init_model_path)
            self._steps = int(init_model_path[init_model_path.rfind('-')+1:])
        if torch.cuda.device_count() > 1:
            self._q_network = nn.DataParallel(self._q_network)
        if use_gpu:
            self._q_network.cuda()
        self._optimizer = optim.RMSprop(
            self._q_network.parameters(), lr=rmsprop_lr,
            eps=rmsprop_eps, centered=False)
        self._target_q_network = deepcopy(self._q_network)

        self._memory = ReplayMemory(memory_size)

    def step(self, observation, greedy=False):
        if greedy or random.uniform(0, 1) >= self._epsilon:
            observation = tuple(torch.from_numpy(np.expand_dims(array, 0))
                                for array in observation)
            if self._use_gpu:
                observation = tuple(tensor.cuda() for tensor in observation)
            q_value = self._q_network(
                tuple(Variable(tensor, volatile=True) for tensor in observation))
            _, action = q_value[0].data.max(0)
            return [action[0]]
        else:
            return [random.randint(0, self._action_spec[0] - 1)]

    def train(self, env, env_fn, num_processes=16, num_episodes=100000000):
        self._init_experience_collector(env_fn, num_processes)
        t = time.time()
        loss_sum = 0.0
        while True:
            while self._queue.qsize() > 0:
                self._memory.push(*(self._queue.get()))
            if len(self._memory) < self._batch_size * 10:
                time.sleep(0.1)
                continue
            if self._steps % self._target_update_freq == 0:
                self._target_q_network = deepcopy(self._q_network)
            if self._epsilon > self._epsilon_min:
                self._epsilon -= self._epsilon_decrease

            loss_sum += self._update()

            if self._steps % self._save_model_freq == 0:
                self._save_model(os.path.join(
                    self._save_model_dir, 'agent.model-%d' % self._steps))
            if self._steps % self._print_freq == 0:
                print("Steps: %d Time: %f Epsilon: %f Loss %f Experiences %d" %
                      (self._steps, time.time() - t, self._epsilon,
                       loss_sum / self._print_freq, len(self._memory)))
                loss_sum = 0.0
                t = time.time()

    def _init_experience_collector(self, env_fn, num_processes):
        #self._queue = multiprocessing.SimpleQueue()
        self._queue = multiprocessing.Queue(100)
        self._processes = [multiprocessing.Process(target=self._collect_experience,
                                                   args=(env_fn, process_id))
                           for process_id in xrange(num_processes)]
        for process in self._processes:
            process.daemon = True
            process.start()

    def _collect_experience(self, env_create_func, process_id):
        env = env_create_func()
        episode = 0
        while True:
            episode += 1
            cum_return = 0.0
            observation, _ = env.reset()
            done = False
            while not done:
                action = self.step(observation)
                next_observation, reward, done, _ = env.step(action)
                self._queue.put((observation, action, reward,
                                 next_observation, done))
                observation = next_observation
                cum_return += reward
            print("ProcessID %d Episode %d Return: %f." %
                  (process_id, episode, cum_return))

    def _update(self):
        if len(self._memory) < self._batch_size:
            return
        transitions = self._memory.sample(self._batch_size)
        (next_observation_batch, observation_batch, reward_batch,
         action_batch, done_batch) = self._transitions_to_batch(transitions)
        # convert to torch variable
        next_observation_batch = tuple(Variable(tensor, volatile=True)
                                       for tensor in next_observation_batch)
        observation_batch = tuple(Variable(tensor)
                                  for tensor in observation_batch)
        reward_batch = Variable(reward_batch)
        action_batch = Variable(action_batch)
        done_batch = Variable(done_batch)
        # compute max-q target
        q_values_next = self._q_network(next_observation_batch)
        q_values_target = self._target_q_network(next_observation_batch)
        futures = q_values_target.gather(
            1, q_values_next.max(dim=1)[1].view(-1, 1)).squeeze()
        futures = futures * (1 - done_batch)
        target_q = reward_batch + self._discount * futures
        target_q.volatile = False
        # compute gradient
        q_values = self._q_network(observation_batch)
        """
        print(torch.cat([q_values.gather(1, action_batch.view(-1, 1)),
                         target_q.unsqueeze(1),
                         action_batch.float(),
                         done_batch.unsqueeze(1)],1))
        """
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q_values.gather(1, action_batch.view(-1, 1)), target_q)
        self._optimizer.zero_grad()
        loss.backward()
        # update q-network
        self._optimizer.step()
        self._steps += 1
        return loss.data[0]

    def _transitions_to_batch(self, transitions):
        batch = Transition(*zip(*transitions))
        next_observation_batch = [torch.from_numpy(np.stack(feat))
                                  for feat in zip(*batch.next_observation)]
        observation_batch = [torch.from_numpy(np.stack(feat))
                             for feat in zip(*batch.observation)]
        reward_batch = torch.FloatTensor(batch.reward)
        action_batch = torch.LongTensor(batch.action)
        done_batch = torch.Tensor(batch.done)
        if self._use_gpu:
            next_observation_batch = [tensor.cuda()
                                      for tensor in next_observation_batch]
            observation_batch = [tensor.cuda() for tensor in observation_batch]
            reward_batch = reward_batch.cuda()
            action_batch = action_batch.cuda()
            done_batch = done_batch.cuda()
        return (next_observation_batch, observation_batch, reward_batch,
                action_batch, done_batch)
                
    def _save_model(self, model_path):
        torch.save(self._q_network.state_dict(), model_path)

    def _load_model(self, model_path):
        self._q_network.load_state_dict(
            torch.load(model_path, map_location=lambda storage, loc: storage))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


class FullyConvNetTiny(nn.Module):
    def __init__(self,
                 resolution,
                 in_channels_screen,
                 in_channels_minimap,
                 out_dims,
                 enable_batchnorm=False):
        super(FullyConvNetTiny, self).__init__()
        self.fc = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, out_dims)

    def forward(self, x):
        screen, minimap, player = x
        x = self.fc3(F.leaky_relu(self.fc2(F.leaky_relu(self.fc(player)))))
        return x


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
        self.q_fc = nn.Linear(256, out_dims)
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
        q = self.q_fc(state)
        return q
