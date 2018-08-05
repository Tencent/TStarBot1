from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from threading import Thread
import os

import torch
from absl import app
from absl import flags
from absl import logging
from memoire import ReplayMemoryServer
from memoire import Bind
from memoire import Conn

from sc2learner.agents.ppo_policies import LstmPolicy
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers import ZergNonspatialObservationWrapper
from sc2learner.agents.dist_dqn_agent import DistRolloutWorker
from sc2learner.agents.dist_dqn_agent import DistDDQNLearner
from sc2learner.agents.q_networks import NonspatialDuelingQNet
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job_name", 'actor', ['actor', 'learner'], "Job type.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_integer("client_memory_size", 50000,
                     "Total size of client memory.")
flags.DEFINE_integer("warmup_size", 10000000, "Warmup size for replay memory.")
flags.DEFINE_integer("cache_size", 128, "Cache size.")
flags.DEFINE_integer("num_caches", 4096, "Number of server caches.")
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
flags.DEFINE_integer("eps_decay2", 10000000, "Greedy epsilon decay step.")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate.")
flags.DEFINE_float("adam_eps", 1e-7, "Adam optimizer's epsilon.")
flags.DEFINE_float("gradient_clipping", 10.0, "Gradient clipping threshold.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("mmc_discount", 0.995, "Discount.")
flags.DEFINE_float("mmc_beta", 0.9, "Discount.")
flags.DEFINE_integer("target_update_freq", 10000, "Target net update frequency.")
flags.DEFINE_string("init_checkpoint_path", "", "Checkpoint to initialize model.")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("checkpoint_freq", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_freq", 10000, "Print train cost frequency.")
flags.DEFINE_boolean("use_curriculum", False, "Use curriculum or not.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.FLAGS(sys.argv)


def tf_config():
  ncpu = multiprocessing.cpu_count()
  if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()


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
  env = ZergActionWrapper(env, mask=False)
  env = ZergNonspatialObservationWrapper(env)
  return env


def start_actor():
  difficulty = random.choice(FLAGS.difficulties.split(','))
  game_seed =  random.randint(0, 2**32 - 1)
  env = create_env(difficulty, game_seed)
  actor = ppo3.PPOActor(env=env,
                        policy=LstmPolicy,
                        unroll_length=128,
                        gamma=0.99,
                        lam=0.95,
                        learner_ip="localhost")
  actor.run()


def start_learner():
  env = create_env(FLAGS.difficulties, 0)
  learner = ppo3.PPOLearner(env=env,
                            policy=LstmPolicy,
                            unroll_length=128,
                            lr=2.5e-4,
                            clip_range=0.1,
                            batch_size=2,
                            print_interval=100)
  learner.run()


def main(argv):
  logging.set_verbosity(logging.ERROR)
  print_arguments(FLAGS)
  tf_config()
  if FLAGS.job_name == 'actor': start_actor()
  else: start_learner()


if __name__ == '__main__':
  app.run(main)
