import sys
import torch
import os
import traceback
from absl import app
from absl import flags

from envs.sc2_env_unit_control import StarCraftIIEnv
from wrappers.zerg_action_unit_control_wrappers import ZergActionWrapper
from wrappers.sc2_observation_wrappers import SC2ObservationWrapper
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.fast_dqn_agent import FastDQNAgent
from models.sc2_networks import SC2DuelingQNetV2

UNIT_TYPE_WHITELIST_TINY = [0, 86, 483, 341, 342, 88, 638, 104, 110, 106,
                            89, 105, 90, 126, 100, 472, 641, 137, 97, 96,
                            103, 107, 98, 688, 108, 129, 99, 9, 91, 151]

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 200, "Number of episodes to evaluate.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '2',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("observation_filter", "effects,player_id",
                    "Observation field to ignore.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent", 'fast_double_dqn',
                  ['fast_dqn', 'fast_double_dqn', 'random', 'keyboard'],
                   "Algorithm.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("render", True, "Visualize feature map or not.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_boolean("flip_minimap_screen", True, "Use batchnorm or not.")
flags.FLAGS(sys.argv)


def create_env():
    env = StarCraftIIEnv(
        map_name='AbyssalReef',
        step_mul=FLAGS.step_mul,
        disable_fog=FLAGS.disable_fog,
        resolution=32,
        agent_race='Z',
        bot_race='Z',
        difficulty=FLAGS.difficulty,
        game_steps_per_episode=0,
        visualize_feature_map=FLAGS.render,
        score_index=None)
    env = ZergActionWrapper(env)
    env = SC2ObservationWrapper(
        env=env,
        unit_type_whitelist=UNIT_TYPE_WHITELIST_TINY,
        observation_filter=FLAGS.observation_filter.split(','),
        flip=FLAGS.flip_minimap_screen)
    return env


def train():
    env = create_env()
    network = SC2DuelingQNetV2(
        resolution=env.observation_space.spaces[0].shape[1],
        n_channels_screen=env.observation_space.spaces[0].shape[0],
        n_channels_minimap=env.observation_space.spaces[1].shape[0],
        n_out=env.action_space.n,
        batchnorm=FLAGS.use_batchnorm)

    if FLAGS.agent == 'fast_dqn' or FLAGS.agent == 'fast_double_dqn':
        agent = FastDQNAgent(
            observation_space=env.observation_space,
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
            eps_decay=1000000,
            memory_size=1000000,
            init_memory_size=100000,
            frame_step_ratio=1.0,
            gradient_clipping=1.0,
            double_dqn=True if FLAGS.agent == 'fast_double_dqn' else False,
            target_update_freq=10000,
            init_model_path=FLAGS.init_model_path)
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    try:
        cum_return = 0.0
        for i in range(FLAGS.num_episodes):
            observation = env.reset()
            done = False
            while not done:
                action = agent.act(observation, eps=0)
                observation, reward, done, _ = env.step(action)
                cum_return += reward
            print("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f" %
                  (i + 1, FLAGS.num_episodes, cum_return / (i + 1),
                   ((cum_return / (i + 1)) + 1) / 2.0))
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()
    env.close()


def main(argv):
    train()


if __name__ == '__main__':
    app.run(main)
