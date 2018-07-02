from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import traceback

import torch
from absl import app
from absl import flags

from sc2learner.envs.sc2_env import StarCraftIIEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers import ZergObservationWrapper
from sc2learner.envs.observations.zerg_observation_wrappers import ZergNonspatialObservationWrapper
from sc2learner.envs.rewards.reward_wrappers import RewardShapingWrapperV2
from sc2learner.agents.dqn_agent import DDQNAgent
from sc2learner.agents.models.sc2_networks import DuelingQNet
from sc2learner.agents.models.sc2_networks import NonspatialDuelingQNet
from sc2learner.agents.models.sc2_networks import NonspatialDuelingLinearQNet
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_integer("num_actor_workers", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulty", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_float("winning_rate_threshold", 0.65, "Winning rate threshold.")
flags.DEFINE_integer("memory_size", 5000000, "Experience replay size.")
flags.DEFINE_integer("init_memory_size", 500000, "Experience replay init size.")
flags.DEFINE_enum("eps_method", 'linear', ['exponential', 'linear'], "Epsilon decay methods.")
flags.DEFINE_float("eps_start", 1.0, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.1, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay", 1000000, "Greedy epsilon decay step.")
flags.DEFINE_integer("eps_decay2", 50000000, "Greedy epsilon decay step.")
flags.DEFINE_enum("optimizer_type", 'adam', ['rmsprop', 'adam', 'sgd'], "Optimizer.")
flags.DEFINE_float("learning_rate", 3e-7, "Learning rate.")
flags.DEFINE_float("momentum", 0.9, "Momentum.")
flags.DEFINE_float("adam_eps", 1e-7, "Adam optimizer's epsilon.")
flags.DEFINE_float("gradient_clipping", 10.0, "Gradient clipping threshold.")
flags.DEFINE_float("frame_step_ratio", 1.0, "Actor frames per train step.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_float("discount", 0.995, "Discount.")
flags.DEFINE_float("mmc_discount", 0.995, "Discount.")
flags.DEFINE_float("mmc_beta", 0.9, "Discount.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_enum("loss_type", 'mse', ['mse', 'smooth_l1'], "Loss type.")
flags.DEFINE_integer("target_update_freq", 10000, "Target net update frequency.")
flags.DEFINE_integer("save_model_freq", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_freq", 10000, "Print train cost frequency.")
flags.DEFINE_boolean("use_action_mask", False, "Use action mask or not.")
flags.DEFINE_boolean("use_curriculum", False, "Use curriculum or not.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("flip_features", True, "Flip 2D features.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_reward_shaping", False, "Enable reward shaping.")
flags.DEFINE_boolean("use_spatial_features", False, "Use spatial features.")
flags.DEFINE_boolean("use_nonlinear_model", True, "Use Nonlinear model.")
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
  if FLAGS.use_reward_shaping: env = RewardShapingWrapperV2(env)
  env = ZergActionWrapper(env, mask=FLAGS.use_action_mask)
  if FLAGS.use_spatial_features:
    env = ZergObservationWrapper(env, flip=FLAGS.flip_features)
  else:
    env = ZergNonspatialObservationWrapper(env)
  return env


def create_network(env):
  if FLAGS.use_spatial_features:
    assert FLAGS.use_nonlinear_model
    network = DuelingQNet(resolution=env.observation_space.spaces[0].shape[1],
                          n_channels=env.observation_space.spaces[0].shape[0],
                          n_dims=env.observation_space.spaces[1].shape[0],
                          n_out=env.action_space.n,
                          batchnorm=FLAGS.use_batchnorm)
  else:
    if FLAGS.use_nonlinear_model:
      network = NonspatialDuelingQNet(n_dims=env.observation_space.shape[0],
                                      n_out=env.action_space.n)
    else:
      network = NonspatialDuelingLinearQNet(
          n_dims=env.observation_space.shape[0],
          n_out=env.action_space.n)
  return network


def train():
  if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
    os.makedirs(FLAGS.save_model_dir)

  env = create_env('1', 0)
  network = create_network(env)

  agent = DDQNAgent(observation_space=env.observation_space,
                    action_space=env.action_space,
                    network=network,
                    optimizer_type=FLAGS.optimizer_type,
                    learning_rate=FLAGS.learning_rate,
                    momentum=FLAGS.momentum,
                    adam_eps=FLAGS.adam_eps,
                    batch_size=FLAGS.batch_size,
                    discount=FLAGS.discount,
                    eps_method=FLAGS.eps_method,
                    eps_start=FLAGS.eps_start,
                    eps_end=FLAGS.eps_end,
                    eps_decay=FLAGS.eps_decay,
                    eps_decay2=FLAGS.eps_decay2,
                    memory_size=FLAGS.memory_size,
                    init_memory_size=FLAGS.init_memory_size,
                    frame_step_ratio=FLAGS.frame_step_ratio,
                    gradient_clipping=FLAGS.gradient_clipping,
                    double_dqn=True,
                    target_update_freq=FLAGS.target_update_freq,
                    winning_rate_threshold=FLAGS.winning_rate_threshold,
                    difficulties=FLAGS.difficulty.strip().split(','),
                    mmc_beta=FLAGS.mmc_beta,
                    mmc_discount=FLAGS.mmc_discount,
                    allow_eval_mode=True,
                    loss_type=FLAGS.loss_type,
                    init_model_path=FLAGS.init_model_path,
                    save_model_dir=FLAGS.save_model_dir,
                    save_model_freq=FLAGS.save_model_freq,
                    print_freq=FLAGS.print_freq)
  try: agent.learn(create_env, FLAGS.num_actor_workers, FLAGS.use_curriculum)
  except KeyboardInterrupt: pass
  except: traceback.print_exc()
  env.close()


def main(argv):
  print_arguments(FLAGS)
  train()


if __name__ == '__main__':
  app.run(main)
