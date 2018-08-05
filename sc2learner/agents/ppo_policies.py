import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input


class LstmPolicy(object):

  def __init__(self, sess, ob_space, ac_space, nbatch, unroll_length, nlstm=256, reuse=False):
    nenv = nbatch // unroll_length
    self.pdtype = make_pdtype(ac_space)
    X, processed_x = observation_input(ob_space, nbatch)

    M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
    S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
    with tf.variable_scope("model", reuse=reuse):
      fc1 = tf.nn.relu(fc(X, 'fc1', 512))
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
