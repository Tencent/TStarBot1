import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input


class LstmPolicy(object):

  def __init__(self, sess, ob_space, ac_space, nbatch, unroll_length, nlstm=256,
               reuse=False):
    nenv = nbatch // unroll_length
    self.pdtype = make_pdtype(ac_space)
    X, processed_x = observation_input(ob_space, nbatch)

    M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
    S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
    with tf.variable_scope("model", reuse=reuse):
      processed_x = tf.layers.flatten(processed_x)
      fc1 = tf.nn.relu(fc(processed_x, 'fc1', 512))
      h = tf.nn.relu(fc(fc1, 'fc2', 512))
      xs = batch_to_seq(h, nenv, unroll_length)
      ms = batch_to_seq(M, nenv, unroll_length)
      h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
      h5 = seq_to_batch(h5)
      vf = fc(h5, 'v', 1)
      self.pd, self.pi = self.pdtype.pdfromlatent(h5)

    v0 = vf[:, 0]
    a0 = self.pd.sample()
    neglogp0 = self.pd.neglogp(a0)
    self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

    def step(ob, state, mask):
      return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

    def value(ob, state, mask):
      return sess.run(v0, {X:ob, S:state, M:mask})

    self.X = X
    self.M = M
    self.S = S
    self.vf = vf
    self.step = step
    self.value = value


class MlpPolicy(object):
  def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
    self.pdtype = make_pdtype(ac_space)
    with tf.variable_scope("model", reuse=reuse):
      X, processed_x = observation_input(ob_space, nbatch)
      activ = tf.tanh
      processed_x = tf.layers.flatten(processed_x)
      pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=256, init_scale=np.sqrt(2)))
      pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=256, init_scale=np.sqrt(2)))
      pi_h3 = activ(fc(pi_h2, 'pi_fc3', nh=256, init_scale=np.sqrt(2)))
      vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=256, init_scale=np.sqrt(2)))
      vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=256, init_scale=np.sqrt(2)))
      vf_h3 = activ(fc(vf_h2, 'vf_fc3', nh=256, init_scale=np.sqrt(2)))
      vf = fc(vf_h3, 'vf', 1)[:,0]

      self.pd, self.pi = self.pdtype.pdfromlatent(pi_h3, init_scale=0.01)

    a0 = self.pd.sample()
    neglogp0 = self.pd.neglogp(a0)
    self.initial_state = None

    def step(ob, *_args, **_kwargs):
      a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
      return a, v, self.initial_state, neglogp

    def value(ob, *_args, **_kwargs):
      return sess.run(vf, {X:ob})

    self.X = X
    self.vf = vf
    self.step = step
    self.value = value
