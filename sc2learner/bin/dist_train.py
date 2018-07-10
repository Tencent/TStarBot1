from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from threading import Thread

import torch
from absl import app
from absl import flags
from absl import logging
from memoire import ReplayMemoryServer
from memoire import Bind
from memoire import Conn

from sc2learner.envs.sc2_env import StarCraftIIEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers import ZergNonspatialObservationWrapper
from sc2learner.agents.dist_dqn_agent import DistRolloutWorker
from sc2learner.agents.dist_dqn_agent import DistDDQNLearner
from sc2learner.agents.models.sc2_networks import NonspatialDuelingQNet
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job", 'actor', ['actor', 'learner'], "Job type.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_integer("client_memory_size", 10000,
                     "Total size of client memory.")
flags.DEFINE_integer("client_memory_warmup_size", 1000,
                     "Warmup size of client memroy.")
flags.DEFINE_integer("cache_size", 1024, "Cache size.")
flags.DEFINE_integer("num_caches", 256, "Number of server caches.")
flags.DEFINE_integer("num_pull_workers", 16, "Number of pull worker for server.")
flags.DEFINE_float("discount", 0.995, "Discount factor.")
flags.DEFINE_float("push_freq", 4.0, "Probability of a step being pushed.")
flags.DEFINE_float("priority_exponent", 0.0, "Exponent for priority sampling.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_integer("memory_size", 5000000, "Experience replay size.")
flags.DEFINE_integer("init_memory_size", 500000, "Experience replay init size.")
flags.DEFINE_float("eps_start", 1.0, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.1, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay", 1000000, "Greedy epsilon decay step.")
flags.DEFINE_integer("eps_decay2", 50000000, "Greedy epsilon decay step.")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate.")
flags.DEFINE_float("adam_eps", 1e-7, "Adam optimizer's epsilon.")
flags.DEFINE_float("gradient_clipping", 10.0, "Gradient clipping threshold.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("mmc_discount", 0.995, "Discount.")
flags.DEFINE_float("mmc_beta", 0.9, "Discount.")
flags.DEFINE_integer("target_update_freq", 10000, "Target net update frequency.")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("checkpoint_freq", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_freq", 10000, "Print train cost frequency.")
flags.DEFINE_boolean("use_curriculum", False, "Use curriculum or not.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.FLAGS(sys.argv)


def create_env(difficulty, random_seed=None):
  env = StarCraftIIEnv(map_name='AbyssalReef',
                       step_mul=FLAGS.step_mul,
                       resolution=16,
                       agent_race='Z',
                       bot_race='Z',
                       difficulty=difficulty,
                       disable_fog=FLAGS.disable_fog,
                       game_steps_per_episode=0,
                       visualize_feature_map=False,
                       score_index=None,
                       random_seed=random_seed)
  env = ZergActionWrapper(env, mask=False)
  env = ZergNonspatialObservationWrapper(env)
  return env


def create_network(env):
  return NonspatialDuelingQNet(n_dims=env.observation_space.shape[0],
                               n_out=env.action_space.n)


def start_learner_job():
  env = create_env('1', 0)
  network = create_network(env)
  agent = DistDDQNLearner(network=network,
                          observation_space=env.observation_space,
                          action_space=env.action_space,
                          num_caches=FLAGS.num_caches,
                          cache_size=FLAGS.cache_size,
                          num_pull_workers=FLAGS.num_pull_workers,
                          discount=FLAGS.discount,
                          priority_exponent=FLAGS.priority_exponent)
  env.close()
  agent.learn(batch_size=FLAGS.batch_size,
              mmc_beta=FLAGS.mmc_beta,
              gradient_clipping=FLAGS.gradient_clipping,
              adam_eps=FLAGS.adam_eps,
              learning_rate=FLAGS.learning_rate,
              target_update_freq=FLAGS.target_update_freq,
              checkpoint_dir=FLAGS.checkpoint_dir,
              checkpoint_freq=FLAGS.checkpoint_freq,
              print_freq=FLAGS.print_freq)


def start_actor_job():
  env = create_env('1', 0)
  network = create_network(env)
  worker = DistRolloutWorker(memory_size=FLAGS.client_memory_size,
                             memory_warmup_size=FLAGS.client_memory_warmup_size,
                             difficulties=FLAGS.difficulties.split(','),
                             env_create_fn=create_env,
                             network=network,
                             action_space=env.action_space,
                             eps_start=FLAGS.eps_start,
                             eps_end=FLAGS.eps_end,
                             eps_decay=FLAGS.eps_decay,
                             eps_decay2=FLAGS.eps_decay2,
                             push_freq=FLAGS.push_freq,
                             learner_ip=FLAGS.learner_ip)
  env.close()
  env = None
  worker.run()


def main(argv):
  logging.set_verbosity(logging.ERROR)
  print_arguments(FLAGS)
  if FLAGS.job == 'actor': start_actor_job()
  else: start_learner_job()


if __name__ == '__main__':
  app.run(main)
