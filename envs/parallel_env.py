import numpy as np
from multiprocessing import Process, Pipe

import gym


def worker(pipe, env_create_func):
    env = env_create_func()
    while True:
        cmd, data = pipe.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done: ob, info = env.reset()
            pipe.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob, info = env.reset()
            pipe.send((ob, info))
        elif cmd == 'close':
            env.close()
            break
        elif cmd == 'get_action_spec':
            pipe.send(env.action_spec)
        elif cmd == 'get_observation_spec':
            pipe.send(env.observation_spec)
        else:
            raise NotImplementedError


class ParallelEnvWrapper(gym.Env):
    def __init__(self, env_create_funcs):
        self._pipes, self._pipes_remote = zip(
            *[Pipe() for _ in range(len(env_create_funcs))])
        self._processes = [Process(target=worker, args=(pipe, env_fn,))
                           for (pipe, env_fn) in zip(self._pipes_remote,
                                                     env_create_funcs)]
        for p in self._processes:
            p.daemon = True
            p.start()

    @property
    def action_spec(self):
        self._pipes[0].send(('get_action_spec', None))
        return self._pipes[0].recv()

    @property
    def observation_spec(self):
        self._pipes[0].send(('get_observation_spec', None))
        return self._pipes[0].recv()

    def _step(self, actions):
        for pipe, action in zip(self._pipes, actions):
            pipe.send(('step', action))
        results = [pipe.recv() for pipe in self._pipes]
        obs, rewards, dones, infos = zip(*results)
        if isinstance(obs[0], tuple):
            n = len(obs[0])
            obs = tuple(np.stack([ob[c] for ob in obs]) for c in xrange(n))
        else:
            obs = np.stack(obs)
        return obs, np.stack(rewards), np.stack(dones), infos

    def _reset(self):
        for pipe in self._pipes:
            pipe.send(('reset', None))
        results = [pipe.recv() for pipe in self._pipes]
        obs, infos = zip(*results)
        if isinstance(obs[0], tuple):
            n = len(obs[0])
            obs = tuple(np.stack([ob[c] for ob in obs]) for c in xrange(n))
        else:
            obs = np.stack(obs)
        return obs, infos

    def _close(self):
        for pipe in self._pipes:
            pipe.send(('close', None))
        for p in self._processes:
            p.join()

    @property
    def num_envs(self):
        return len(self._pipes)
