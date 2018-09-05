from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import traceback
import multiprocessing
import time

import torch
from absl import app
from absl import flags

from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers import ZergObservationWrapper
from sc2learner.envs.rewards.reward_wrappers import RewardShapingWrapperV2
from sc2learner.agents.random_agent import RandomAgent
from sc2learner.agents.keyboard_agent import KeyboardAgent
from sc2learner.agents.dqn_agent import DDQNAgent
from sc2learner.agents.dqn_networks import DuelingQNet
from sc2learner.agents.dqn_networks import NonspatialDuelingQNet
from sc2learner.utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_parallels", 4, "Parallel number.")
flags.DEFINE_integer("num_episodes", 50, "Number of episodes to evaluate.")
flags.DEFINE_float("epsilon", 0.0, "Epsilon for policy.")
flags.DEFINE_string("game_version", '4.1.2', "Game core version.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent", 'dqn', ['dqn', 'random', 'keyboard'], "Algorithm.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_action_mask", False, "Use action mask or not.")
flags.DEFINE_boolean("use_spatial_features", False, "Use spatial features.")
flags.DEFINE_boolean("use_region_features", True, "Use region features")
flags.FLAGS(sys.argv)


def create_env(random_seed=None):
  env = SC2RawEnv(map_name='AbyssalReef',
                  step_mul=FLAGS.step_mul,
                  disable_fog=FLAGS.disable_fog,
                  resolution=16,
                  agent_race='zerg',
                  bot_race='zerg',
                  difficulty=FLAGS.difficulty,
                  random_seed=random_seed)
  env = ZergActionWrapper(env,
                          game_version=FLAGS.game_version,
                          mask=FLAGS.use_action_mask,
                          use_all_combat_actions=FLAGS.use_all_combat_actions)
  env = ZergObservationWrapper(env,
                               use_spatial_features=FLAGS.use_spatial_features,
                               use_regions=FLAGS.use_region_features)
  return env


def create_network(env):
  if FLAGS.use_spatial_features:
    network = DuelingQNet(resolution=env.observation_space.spaces[0].shape[1],
                          n_channels=env.observation_space.spaces[0].shape[0],
                          n_dims=env.observation_space.spaces[1].shape[0],
                          n_out=env.action_space.n,
                          batchnorm=FLAGS.use_batchnorm)
  else:
    network = NonspatialDuelingQNet(n_dims=env.observation_space.shape[0],
                                    n_out=env.action_space.n)
  return network


def print_actions(env):
  print("----------------------------- Actions -----------------------------")
  for action_id, action_name in enumerate(env.action_names):
    print("Action ID: %d	Action Name: %s" % (action_id, action_name))
  print("-------------------------------------------------------------------")


def train(pid):
  env = create_env(0)
  print_actions(env)

  if FLAGS.agent == 'dqn':
    network = create_network(env)
    agent = DDQNAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      network=network,
                      optimizer_type='adam',
                      learning_rate=0,
                      momentum=0.95,
                      adam_eps=1e-7,
                      batch_size=128,
                      discount=0.99,
                      eps_method='linear',
                      eps_start=0,
                      eps_end=0,
                      eps_decay=5000000,
                      eps_decay2=30000000,
                      memory_size=1000000,
                      winning_rate_threshold=0,
                      difficulties=[],
                      mmc_beta=0,
                      mmc_discount=0,
                      init_memory_size=100000,
                      frame_step_ratio=1.0,
                      gradient_clipping=1.0,
                      double_dqn=True,
                      target_update_freq=10000,
                      init_model_path=FLAGS.init_model_path)
  elif FLAGS.agent == 'random':
    agent = RandomAgent(action_space=env.action_space)
  elif FLAGS.agent == 'keyboard':
    agent = KeyboardAgent(action_space=env.action_space)
  else:
    raise NotImplementedError
  env.close()

  try:
    cum_return = 0.0
    for i in range(FLAGS.num_episodes):
      random_seed =  int(time.time() * 1000) & 0xFFFFFFFF
      env = create_env(random_seed)
      observation = env.reset()
      done = False
      while not done:
        action = agent.act(observation, eps=FLAGS.epsilon)
        observation, reward, done, _ = env.step(action)
        cum_return += reward
      print("Process: %d Episode: %d Outcome: %f" % (pid, i, reward))
      print("Process: %d Evaluated %d/%d Episodes Avg Return %f "
            "Avg Winning Rate %f" % (pid, i + 1, FLAGS.num_episodes,
                                     cum_return / (i + 1),
                                     ((cum_return / (i + 1)) + 1) / 2.0))
      env.close()
  except KeyboardInterrupt: pass
  except: traceback.print_exc()


def main(argv):
  print_arguments(FLAGS)
  processes = [multiprocessing.Process(target=train, args=(pid,))
               for pid in range(FLAGS.num_parallels)]
  for p in processes:
    p.daemon = True
    p.start()
    time.sleep(1)
  for p in processes:
    p.join()

if __name__ == '__main__':
  app.run(main)
