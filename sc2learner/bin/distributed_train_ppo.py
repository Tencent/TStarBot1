from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from threading import Thread
import os
import multiprocessing
import tensorflow as tf
import random

import torch
from absl import app
from absl import flags
from absl import logging

from sc2learner.agents.ppo_policies import LstmPolicy, MlpPolicy
from sc2learner.agents.ppo_agent import PPOActor, PPOLearner
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_enum("job_name", 'actor', ['actor', 'learner'], "Job type.")
flags.DEFINE_enum("policy", 'lstm', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_integer("unroll_length", 128, "Length of rollout steps.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_string("game_version", '4.1.2', "Game core version.")
flags.DEFINE_float("discount_gamma", 0.995, "Discount factor.")
flags.DEFINE_float("lambda_return", 0.95, "Lambda return factor.")
flags.DEFINE_float("clip_range", 0.1, "Clip range for PPO.")
flags.DEFINE_float("ent_coef", 0.01, "Coefficient for the entropy term.")
flags.DEFINE_float("learn_act_speed_ratio", 0, "Maximum learner/actor ratio.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("learner_queue_size", 128, "Size of learner's unroll queue.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_float("learning_rate", 2.5e-4, "Learning rate.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_interval", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_interval", 100, "Print train cost frequency.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_region_wise_combat", False, "Use region-wise combat.")
flags.DEFINE_boolean("use_action_mask", True, "Use region-wise combat.")
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
                  random_seed=random_seed)
  env = ZergActionWrapper(env,
                          game_version=FLAGS.game_version,
                          mask=FLAGS.use_action_mask,
                          region_wise_combat=FLAGS.use_region_wise_combat)
  env = ZergObservationWrapper(env,
                               use_spatial_features=False,
                               use_game_progress=(not FLAGS.policy == 'lstm'),
                               use_action_seq=(not FLAGS.policy == 'lstm'),
                               divide_regions=FLAGS.use_region_wise_combat)
  return env


def start_actor():
  difficulty = random.choice(FLAGS.difficulties.split(','))
  game_seed =  random.randint(0, 2**32 - 1)
  print("Game Seed: %d" % game_seed)
  env = create_env(difficulty, game_seed)
  policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]
  actor = PPOActor(env=env,
                   policy=policy,
                   unroll_length=FLAGS.unroll_length,
                   gamma=FLAGS.discount_gamma,
                   lam=FLAGS.lambda_return,
                   learner_ip=FLAGS.learner_ip)
  actor.run()
  env.close()


def start_learner():
  env = create_env('1', 0)
  policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]
  learner = PPOLearner(env=env,
                       policy=policy,
                       unroll_length=FLAGS.unroll_length,
                       lr=FLAGS.learning_rate,
                       clip_range=FLAGS.clip_range,
                       batch_size=FLAGS.batch_size,
                       ent_coef=FLAGS.ent_coef,
                       vf_coef=0.5,
                       max_grad_norm=0.5,
                       queue_size=FLAGS.learner_queue_size,
                       print_interval=FLAGS.print_interval,
                       save_interval=FLAGS.save_interval,
                       learn_act_speed_ratio=FLAGS.learn_act_speed_ratio,
                       save_dir=FLAGS.save_dir)
  learner.run()
  env.close()


def main(argv):
  logging.set_verbosity(logging.ERROR)
  print_arguments(FLAGS)
  tf_config()
  if FLAGS.job_name == 'actor': start_actor()
  else: start_learner()


if __name__ == '__main__':
  app.run(main)
