import sys
import torch
import os
import traceback
from absl import app
from absl import flags

from envs.sc2_env import StarCraftIIEnv
from wrappers.terran_action_wrappers import TerranActionWrapperV0
from wrappers.sc2_observation_wrappers import SC2ObservationWrapper
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.dqn_agent import DQNAgent
from agents.fast_dqn_agent import FastDQNAgent
from models.sc2_networks import SC2QNet

UNIT_TYPE_WHITELIST_TINY = [0, 132, 341, 21, 483, 20, 342, 18, 27, 19,
                            45, 28, 638, 47, 48, 22, 32, 38, 23, 472,
                            54, 39, 641, 33, 35, 130, 37, 29, 24, 42,
                            57, 51, 134, 41, 692, 46, 36, 40, 53, 56,
                            52, 268, 55, 49, 30, 5]

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 200, "Number of episodes to evaluate.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("observation_filter", "effects,player_id,creep",
                    "Observation field to ignore.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent", 'fast_double_dqn',
                  ['dqn', 'double_dqn', 'fast_dqn', 'fast_double_dqn',
                   'random', 'keyboard'],
                   "Algorithm.")
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
    env = create_env()
    network = SC2QNet(
        resolution=env.observation_space.spaces[0].shape[1],
        n_channels_screen=env.observation_space.spaces[0].shape[0],
        n_channels_minimap=env.observation_space.spaces[1].shape[0],
        n_out=env.action_space.n,
        batchnorm=FLAGS.use_batchnorm)

    if FLAGS.agent == 'dqn' or FLAGS.agent == 'double_dqn':
        agent = DQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_rate=0,
            momentum=0.95,
            optimize_freq=1,
            batch_size=128,
            discount=0.999,
            eps_method='linear',
            eps_start=0,
            eps_end=0,
            eps_decay=1000000,
            memory_size=1000000,
            gradient_clipping=1.0,
            double_dqn=True if FLAGS.agent == 'double_dqn' else False,
            target_update_freq=10000,
            init_model_path=FLAGS.init_model_path)
    elif FLAGS.agent == 'fast_dqn' or FLAGS.agent == 'fast_double_dqn':
        agent = FastDQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            network=network,
            learning_rate=0,
            momentum=0.95,
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
            print("Evaluated %d/%d Episodes Avg Return %f "
                  "Avg Winning Rate %f" % 
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
