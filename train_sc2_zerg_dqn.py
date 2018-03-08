import sys
import torch
import os
import traceback
from absl import app
from absl import flags

from envs.sc2_env import StarCraftIIEnv
from wrappers.terran_action_wrappers import TerranActionWrapperV0
from wrappers.sc2_observation_wrappers import SC2ObservationWrapper
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from models.sc2_networks import SC2QNet
from utils.utils import print_arguments

UNIT_TYPE_WHITELIST = [0, 86, 483, 341, 342, 88, 638, 104, 110, 106,
                       89, 105, 90, 126, 100, 472, 641, 137, 97, 96,
                       103, 107, 98, 688, 108, 129, 99, 9, 91, 151,
                       94, 502, 503, 101, 92, 8, 112, 504, 87, 138,
                       127, 687, 289, 114, 139, 95, 501, 109, 128, 118,
                       494, 142, 113, 93, 16, 111, 489, 493, 893, 693,
                       102, 140, 499, 7, 119, 892, 17, 115, 12, 150,
                       117, 116, 690, 125]

UNIT_TYPE_WHITELIST_TINY = [0, 86, 483, 341, 342, 88, 638, 104, 110, 106,
                            89, 105, 90, 126, 100, 472, 641, 137, 97, 96,
                            103, 107, 98, 688, 108, 129, 99, 9, 91, 151,
                            94, 502, 503, 101, 92, 8, 112, 504, 87, 138,
                            127, 687, 289, 114, 139, 95, 501, 109, 128, 118,
                            494, 142, 113, 93, 16, 111, 489, 493, 893, 693]

FLAGS = flags.FLAGS
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '2',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("observation_filter", "effects,player_id",
                    "Observation field to ignore.")
flags.DEFINE_integer("memory_size", 50000, "Experience replay size.")
flags.DEFINE_integer("init_memory_size", 2000, "Experience replay initial size.")
flags.DEFINE_enum("eps_method", 'linear', ['eps', 'linear'],
                  "Epsilon decay methods.")
flags.DEFINE_float("eps_start", 1.0, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.1, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay", 100000, "Greedy epsilon decay step.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
flags.DEFINE_float("momentum", 0.95, "Momentum.")
flags.DEFINE_float("gradient_clipping", 1.0, "Gradient clipping threshold.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_float("discount", 0.999, "Discount.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_enum("agent", 'dqn', ['dqn', 'double_dqn'], "Algorithm.")
flags.DEFINE_enum("loss_type", 'mse', ['mse', 'smooth_l1'], "Loss type.")
flags.DEFINE_integer("target_update_freq", 1000, "Target net update frequency.")
flags.DEFINE_integer("optimize_freq", 4, "Frames between two optimizations")
flags.DEFINE_integer("save_model_freq", 50, "Model saving frequency.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("allow_eval_mode", False,
                     "Allow eval() during training, for batchnorm.")
flags.FLAGS(sys.argv)


def create_env():
    env = StarCraftIIEnv(
        map_name='AbyssalReef',
        step_mul=FLAGS.step_mul,
        resolution=32,
        agent_race='Z',
        bot_race='Z',
        difficulty=FLAGS.difficulty,
        game_steps_per_episode=0,
        visualize_feature_map=False,
        score_index=None)
    env = TerranActionWrapperV0(env)
    env = SC2ObservationWrapper(
        env=env,
        unit_type_whitelist=UNIT_TYPE_WHITELIST_TINY,
        observation_filter=FLAGS.observation_filter.split(','))
    return env


def train():
    if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
        os.makedirs(FLAGS.save_model_dir)

    env = create_env()
    network = SC2QNet(
        resolution=env.observation_space.spaces[0].shape[1],
        n_channels_screen=env.observation_space.spaces[0].shape[0],
        n_channels_minimap=env.observation_space.spaces[1].shape[0],
        n_out=env.action_space.n,
        batchnorm=FLAGS.use_batchnorm)

    if FLAGS.agent == 'dqn':
        agent = DQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_rate=FLAGS.learning_rate,
            momentum=FLAGS.momentum,
            optimize_freq=FLAGS.optimize_freq,
            batch_size=FLAGS.batch_size,
            discount=FLAGS.discount,
            eps_method=FLAGS.eps_method,
            eps_start=FLAGS.eps_start,
            eps_end=FLAGS.eps_end,
            eps_decay=FLAGS.eps_decay,
            memory_size=FLAGS.memory_size,
            init_memory_size=FLAGS.init_memory_size,
            gradient_clipping=FLAGS.gradient_clipping,
            allow_eval_mode=FLAGS.allow_eval_mode,
            loss_type=FLAGS.loss_type,
            init_model_path=FLAGS.init_model_path,
            save_model_dir=FLAGS.save_model_dir,
            save_model_freq=FLAGS.save_model_freq)
    elif FLAGS.agent == 'double_dqn':
        agent = DoubleDQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_rate=FLAGS.learning_rate,
            momentum=FLAGS.momentum,
            optimize_freq=FLAGS.optimize_freq,
            batch_size=FLAGS.batch_size,
            discount=FLAGS.discount,
            eps_method=FLAGS.eps_method,
            eps_start=FLAGS.eps_start,
            eps_end=FLAGS.eps_end,
            eps_decay=FLAGS.eps_decay,
            memory_size=FLAGS.memory_size,
            init_memory_size=FLAGS.init_memory_size,
            gradient_clipping=FLAGS.gradient_clipping,
            target_update_freq=FLAGS.target_update_freq,
            allow_eval_mode=FLAGS.allow_eval_mode,
            loss_type=FLAGS.loss_type,
            init_model_path=FLAGS.init_model_path,
            save_model_dir=FLAGS.save_model_dir,
            save_model_freq=FLAGS.save_model_freq)
    else:
        raise NotImplementedError

    try:
        agent.learn(env)
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()
    env.close()


def main(argv):
    print_arguments(FLAGS)
    train()


if __name__ == '__main__':
    app.run(main)
