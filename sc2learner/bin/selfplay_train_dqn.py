from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from threading import Thread
import os
import time

import torch
from absl import app
from absl import flags
from absl import logging
from memoire import ReplayMemoryServer
from memoire import Bind
from memoire import Conn

from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.selfplay_raw_env import SC2SelfplayRawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers import ZergNonspatialObservationWrapper
from sc2learner.agents.selfplay_dist_dqn_agent import SelfplayDistRolloutWorker
from sc2learner.agents.selfplay_dist_dqn_agent import DistRolloutWorker
from sc2learner.agents.selfplay_dist_dqn_agent import DistDDQNLearner
from sc2learner.agents.q_networks import NonspatialDuelingQNet
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job", 'actor', ['actor', 'learner', 'eval'], "Job type.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_string("game_version", '4.1.2', "Game core version.")
flags.DEFINE_integer("client_memory_size", 50000,
                     "Total size of client memory.")
flags.DEFINE_integer("warmup_size", 5000000, "Warmup size for replay memory.")
flags.DEFINE_integer("cache_size", 128, "Cache size.")
flags.DEFINE_integer("num_caches", 4096, "Number of server caches.")
flags.DEFINE_integer("num_pull_workers", 16, "Number of pull worker for server.")
flags.DEFINE_float("discount", 0.995, "Discount factor.")
flags.DEFINE_float("push_freq", 8.0, "Probability of a step being pushed.")
flags.DEFINE_float("priority_exponent", 0.0, "Exponent for priority sampling.")
flags.DEFINE_integer("model_cache_size", 200, "Opponent model cache size.")
flags.DEFINE_float("model_cache_prob", 0.02, "Opponent model cache probability.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_float("eps_start", 1.0, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.1, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay", 3000000, "Greedy epsilon decay step.")
flags.DEFINE_integer("eps_decay2", 10000000, "Greedy epsilon decay step.")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate.")
flags.DEFINE_float("adam_eps", 1e-7, "Adam optimizer's epsilon.")
flags.DEFINE_float("gradient_clipping", 10.0, "Gradient clipping threshold.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_float("mmc_discount", 0.995, "Discount.")
flags.DEFINE_float("mmc_beta", 0.9, "Discount.")
flags.DEFINE_integer("target_update_freq", 10000, "Target net update frequency.")
flags.DEFINE_string("init_checkpoint_path", "", "Checkpoint to initialize model.")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("checkpoint_freq", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_freq", 10000, "Print train cost frequency.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.FLAGS(sys.argv)


def create_env(difficulty, random_seed=None):
  env = SC2RawEnv(map_name='AbyssalReef',
                  step_mul=FLAGS.step_mul,
                  resolution=16,
                  agent_race='zerg',
                  bot_race='zerg',
                  difficulty=difficulty,
                  disable_fog=FLAGS.disable_fog,
                  game_steps_per_episode=0,
                  visualize_feature_map=False,
                  random_seed=random_seed)
  env = ZergActionWrapper(env, FLAGS.game_version)
  env = ZergNonspatialObservationWrapper(env)
  return env


def create_selfplay_env(random_seed=None):
  env = SC2SelfplayRawEnv(map_name='AbyssalReef',
                         step_mul=FLAGS.step_mul,
                         resolution=16,
                         agent_race='zerg',
                         opponent_race='zerg',
                         disable_fog=FLAGS.disable_fog,
                         visualize_feature_map=False,
                         random_seed=random_seed)
  env = ZergActionWrapper(env, FLAGS.game_version)
  env = ZergNonspatialObservationWrapper(env)
  return env


def create_network(env):
  return NonspatialDuelingQNet(n_dims=env.observation_space.shape[0],
                               n_out=env.action_space.n)


def start_actor_job():
  random_seed = int(time.time() * 1e6) & 0xFFFFFFFF
  env = create_selfplay_env(random_seed)
  network = create_network(env)
  worker = SelfplayDistRolloutWorker(env=env,
                                     network=network,
                                     memory_size=FLAGS.client_memory_size,
                                     push_freq=FLAGS.push_freq,
                                     model_cache_prob=FLAGS.model_cache_prob,
                                     model_cache_size=FLAGS.model_cache_size,
                                     learner_ip=FLAGS.learner_ip,
                                     game_version=FLAGS.game_version)
  worker.run()
  env.close()


def start_eval_job():
  env = create_env('1', 0)
  network = create_network(env)
  worker = DistRolloutWorker(memory_size=FLAGS.client_memory_size,
                             difficulties=FLAGS.difficulties.split(','),
                             env_create_fn=create_env,
                             network=network,
                             action_space=env.action_space,
                             push_freq=FLAGS.push_freq,
                             learner_ip=FLAGS.learner_ip,
                             enable_push=False)
  env.close()
  env = None
  worker.run()


def start_learner_job():
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  env = create_env('1', 0)
  network = create_network(env)
  agent = DistDDQNLearner(network=network,
                          observation_space=env.observation_space,
                          action_space=env.action_space,
                          num_caches=FLAGS.num_caches,
                          cache_size=FLAGS.cache_size,
                          num_pull_workers=FLAGS.num_pull_workers,
                          eps_start=FLAGS.eps_start,
                          eps_end=FLAGS.eps_end,
                          eps_decay=FLAGS.eps_decay,
                          eps_decay2=FLAGS.eps_decay2,
                          discount=FLAGS.discount,
                          init_checkpoint_path=FLAGS.init_checkpoint_path,
                          priority_exponent=FLAGS.priority_exponent)
  env.close()
  env = None
  agent.learn(batch_size=FLAGS.batch_size,
              mmc_beta=FLAGS.mmc_beta,
              gradient_clipping=FLAGS.gradient_clipping,
              adam_eps=FLAGS.adam_eps,
              warmup_size=FLAGS.warmup_size,
              learning_rate=FLAGS.learning_rate,
              target_update_freq=FLAGS.target_update_freq,
              checkpoint_dir=FLAGS.checkpoint_dir,
              checkpoint_freq=FLAGS.checkpoint_freq,
              print_freq=FLAGS.print_freq)


def main(argv):
  logging.set_verbosity(logging.ERROR)
  print_arguments(FLAGS)
  if FLAGS.job == 'actor': start_actor_job()
  elif FLAGS.job == 'eval': start_eval_job()
  else: start_learner_job()


if __name__ == '__main__':
  app.run(main)
