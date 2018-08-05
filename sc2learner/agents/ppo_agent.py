import os
import time
from collections import deque
from queue import Queue
from threading import Thread
import time
import random

import joblib
import numpy as np
import tensorflow as tf
import zmq

from baselines.common import explained_variance


class Model(object):
  def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
               unroll_length, ent_coef, vf_coef, max_grad_norm):
    sess = tf.get_default_session()

    act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
    train_model = policy(sess, ob_space, ac_space, nbatch_train, unroll_length,
                         reuse=True)

    A = train_model.pdtype.sample_placeholder([None])
    ADV = tf.placeholder(tf.float32, [None])
    R = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    OLDVPRED = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRANGE = tf.placeholder(tf.float32, [])

    neglogpac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    vpred = train_model.vf
    vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                               -CLIPRANGE, CLIPRANGE)
    vf_losses1 = tf.square(vpred - R)
    vf_losses2 = tf.square(vpredclipped - R)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE,
                                         1.0 + CLIPRANGE)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    clipfrac = tf.reduce_mean(
        tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
    with tf.variable_scope('model'):
      params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    if max_grad_norm is not None:
      grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)
    new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
    param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, new_params)]

    def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs,
              states=None):
      advs = returns - values
      advs = (advs - advs.mean()) / (advs.std() + 1e-8)
      td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
      if states is not None:
        td_map[train_model.S] = states
        td_map[train_model.M] = masks
      return sess.run(
        [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
        td_map
      )[:-1]
    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                       'approxkl', 'clipfrac']

    def save(save_path):
      joblib.dump(read_params(), save_path)

    def load(load_path):
      loaded_params = joblib.load(load_path)
      load_params(loaded_params)

    def read_params():
      return sess.run(params)

    def load_params(loaded_params):
      sess.run(param_assign_ops,
               feed_dict={p : v for p, v in zip(new_params, loaded_params)})

    self.train = train
    self.train_model = train_model
    self.act_model = act_model
    self.step = act_model.step
    self.value = act_model.value
    self.initial_state = act_model.initial_state
    self.save = save
    self.load = load
    self.read_params = read_params
    self.load_params = load_params
    tf.global_variables_initializer().run(session=sess)


class PPOActor(object):

  def __init__(self, env, policy, unroll_length, gamma, lam,
               learner_ip="localhost", queue_size=1):
    self._env = env
    self._unroll_length = unroll_length
    self._lam = lam
    self._gamma = gamma

    self._model = Model(policy=policy,
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=unroll_length,
                        unroll_length=unroll_length,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5)
    self._obs = np.zeros(env.observation_space.shape,
                         dtype=env.observation_space.dtype.name)
    self._obs[:] = env.reset()
    self._state = self._model.initial_state
    self._done = False

    self._zmq_context = zmq.Context()
    self._model_requestor = self._zmq_context.socket(zmq.REQ)
    self._model_requestor.connect("tcp://%s:5701" % learner_ip)
    self._data_queue = Queue(queue_size)
    self._push_thread = Thread(target=self._push_data, args=(
        self._zmq_context, learner_ip, self._data_queue))
    self._push_thread.start()

  def run(self):
    while True:
      self._update_model()
      if self._data_queue.full(): print("[WARN]: Actor's queue is full.")
      self._data_queue.put(self._nstep_rollout())

  def _nstep_rollout(self):
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = \
        [],[],[],[],[],[]
    mb_states, episode_infos = self._state, []
    for _ in range(self._unroll_length):
      action, value, self._state, neglogpac = self._model.step(
          np.expand_dims(self._obs, 0), self._state,
          np.expand_dims(self._done, 0))
      mb_obs.append(self._obs.copy())
      mb_actions.append(action[0])
      mb_values.append(value[0])
      mb_neglogpacs.append(neglogpac[0])
      mb_dones.append(self._done)
      self._obs[:], reward, self._done, info = self._env.step(action)
      if self._done:
        self._obs[:] = self._env.reset()
        self._state = self._model.initial_state
      if 'episode' in info: episode_infos.append(info.get('episode'))
      mb_rewards.append(reward)
    mb_obs = np.asarray(mb_obs, dtype=self._obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self._model.value(np.expand_dims(self._obs, 0), self._state,
                                    np.expand_dims(self._done, 0))
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    last_gae_lam = 0
    for t in reversed(range(self._unroll_length)):
      if t == self._unroll_length - 1:
        next_nonterminal = 1.0 - self._done
        next_values = last_values[0]
      else:
        next_nonterminal = 1.0 - mb_dones[t + 1]
        next_values = mb_values[t + 1]
      delta = mb_rewards[t] + self._gamma * next_values * next_nonterminal - \
          mb_values[t]
      mb_advs[t] = last_gae_lam = delta + self._gamma * self._lam * \
          next_nonterminal * last_gae_lam
    mb_returns = mb_advs + mb_values
    return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
            mb_states, episode_infos)

  def _push_data(self, zmq_context, learner_ip, data_queue):
    sender = zmq_context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 1)
    sender.setsockopt(zmq.RCVHWM, 1)
    sender.connect("tcp://%s:5700" % learner_ip)
    while True:
      data = data_queue.get()
      sender.send_pyobj(data)

  def _update_model(self):
      self._model_requestor.send_string("request model")
      self._model.load_params(self._model_requestor.recv_pyobj())


class PPOLearner(object):

  def __init__(self, env, policy, unroll_length, lr, clip_range, batch_size,
               ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, queue_size=8,
               print_interval=100, save_interval=10000, learn_act_speed_ratio=0,
               save_dir=None, load_path=None):
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(clip_range, float): clip_range = constfn(clip_range)
    else: assert callable(cliprange)
    self._lr = lr
    self._clip_range=clip_range
    self._batch_size = batch_size
    self._unroll_length = unroll_length
    self._print_interval = print_interval
    self._save_interval = save_interval
    self._learn_act_speed_ratio = learn_act_speed_ratio
    self._save_dir = save_dir

    self._model = Model(policy=policy,
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=batch_size * unroll_length,
                        unroll_length=unroll_length,
                        ent_coef=ent_coef,
                        vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)
    if load_path is not None: self._model.load(load_path)
    self._model_params = self._model.read_params()
    self._data_queue = deque(maxlen=queue_size)
    self._episode_infos = deque(maxlen=200)
    self._rollout_fps = -1
    self._num_unrolls = 0

    self._zmq_context = zmq.Context()
    self._pull_data_thread = Thread(
        target=self._pull_data,
        args=(self._zmq_context, self._data_queue, self._episode_infos)
    )
    self._pull_data_thread.start()
    self._reply_model_thread = Thread(target=self._reply_model,
                                      args=(self._zmq_context, self._model))
    self._reply_model_thread.start()

  def run(self):
    while len(self._data_queue) < self._data_queue.maxlen: time.sleep(1)
    update, loss = 0, []
    time_start = time.time()
    while True:
      while (self._learn_act_speed_ratio > 0 and
          update * self._batch_size >= \
          self._num_unrolls * self.learn_act_speed_ratio):
        time.sleep(0.001)
      update += 1
      lr_now = self._lr(update)
      clip_range_now = self._clip_range(update)

      batch = random.sample(self._data_queue, self._batch_size)
      obs, returns, dones, actions, values, neglogpacs, states = (
          np.concatenate(arr) if arr[0] is not None else None
          for arr in zip(*batch))
      loss.append(self._model.train(lr_now, clip_range_now, obs, returns, dones,
                                    actions, values, neglogpacs, states))
      self._model_params = self._model.read_params()

      if update % self._print_interval == 0:
        loss_mean = np.mean(loss, axis=0)
        batch_steps = self._batch_size * self._unroll_length
        time_elapsed = time.time() - time_start
        train_fps = self._print_interval * batch_steps / time_elapsed
        rollout_fps = self._print_interval * batch_steps / time_elapsed
        var = explained_variance(values, returns)
        avg_reward = safemean([info['r'] for info in self._episode_infos])
        print("Update: %d	Train-fps: %.1f	Rollout-fps: %.1f	"
              "Explained-var: %.5f	Avg-reward %.2f	Policy-loss: %.5f	"
              "Value-loss: %.5f	Policy-entropy: %.5f	Time-elapsed: %.1f" % (
              update, train_fps, self._rollout_fps, var, avg_reward,
              *loss_mean[:3], time_elapsed))
        time_start, loss = time.time(), []

      if self._save_dir is not None and update % self._save_interval == 0:
        os.makedirs(self._save_dir, exist_ok=True)
        save_path = os.path.join(self._save_dir, 'checkpoints-%i' % update)
        self._model.save(save_path)
        print('Saved to %s.' % save_path)

  def _pull_data(self, zmq_context, data_queue, episode_infos):
    receiver = zmq_context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 1)
    receiver.setsockopt(zmq.SNDHWM, 1)
    receiver.bind("tcp://*:5700")
    start_time, num_frames_now = time.time(), 0
    while True:
      data = receiver.recv_pyobj()
      data_queue.append(data[:-1])
      episode_infos.extend(data[-1])
      self._num_unrolls += 1
      num_frames_now += data[0].shape[0]
      if self._num_unrolls % 100 == 0:
        self._rollout_fps = num_frames_now / (time.time() - start_time)
        start_time, num_frames_now = time.time(), 0

  def _reply_model(self, zmq_context, model):
    receiver = zmq_context.socket(zmq.REP)
    receiver.bind("tcp://*:5701")
    while True:
      msg = receiver.recv_string()
      assert msg == "request model"
      receiver.send_pyobj(self._model_params)


def constfn(val):
  def f(_):
    return val
  return f


def safemean(xs):
  return np.nan if len(xs) == 0 else np.mean(xs)
