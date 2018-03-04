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

UNIT_TYPE_WHITELIST = [0, 5, 6, 11, 18, 19, 20, 21, 22, 23,
                       24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                       44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                       54, 55, 56, 57, 58, 130, 132, 134, 146, 147,
                       149, 268, 341, 342, 343, 365, 472, 473, 474, 483,
                       484, 490, 498, 500, 561, 609, 638, 639, 640, 641,
                       662, 665, 666, 689, 691, 692, 734, 830, 880, 1879,
                       1883]

UNIT_TYPE_WHITELIST_TINY = [0, 132, 341, 21, 483, 20, 342, 18, 27, 19,
                            45, 28, 638, 47, 48, 22, 32, 38, 23, 472,
                            54, 39, 641, 33, 35, 130, 37, 29, 24, 42,
                            57, 51, 134, 41, 692, 46, 36, 40, 53, 56,
                            52, 268, 55, 49, 30, 5]


FLAGS = flags.FLAGS
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("observation_filter", "effects,player_id,creep",
                    "Observation field to ignore.")
flags.DEFINE_integer("memory_size", 10000, "Experience replay size.")
flags.DEFINE_float("eps_start", 0.9, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.05, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay", 2000, "Greedy epsilon decay step.")
flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_float("discount", 0.999, "Discount.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_enum("agent", 'dqn', ['dqn', 'double_dqn'], "Algorithm.")
flags.DEFINE_integer("target_update_freq", 100, "Target net update frequency.")
flags.DEFINE_integer("save_model_freq", 100, "Model saving frequency.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.FLAGS(sys.argv)


def create_env():
    env = StarCraftIIEnv(
        map_name='AbyssalReef',
        step_mul=FLAGS.step_mul,
        resolution=32,
        agent_race='T',
        bot_race='T',
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
        os.mkdir(FLAGS.save_model_dir)

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
            batch_size=FLAGS.batch_size,
            discount=FLAGS.discount,
            eps_start=FLAGS.eps_start,
            eps_end=FLAGS.eps_end,
            eps_decay=FLAGS.eps_decay,
            memory_size=FLAGS.memory_size,
            init_model_path=FLAGS.init_model_path,
            save_model_dir=FLAGS.save_model_dir,
            save_model_freq=FLAGS.save_model_freq)
    elif FLAGS.agent == 'double_dqn':
        agent = DoubleDQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            discount=FLAGS.discount,
            eps_start=FLAGS.eps_start,
            eps_end=FLAGS.eps_end,
            eps_decay=FLAGS.eps_decay,
            memory_size=FLAGS.memory_size,
            target_update_freq=FLAGS.target_update_freq,
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
    train()


if __name__ == '__main__':
    app.run(main)
