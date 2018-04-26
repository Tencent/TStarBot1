import sys
import torch
import os
import traceback
from absl import app
from absl import flags
import numpy as np
import random

from envs.sc2_env import StarCraftIIEnv
from envs.wrappers.zerg_action_wrappers import ZergActionWrapper
from envs.wrappers.zerg_observation_wrappers import ZergObservationWrapper
from envs.wrappers.reward_wrappers import RewardShapingWrapperV2
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.fast_dqn_agent import FastDQNAgent
from agents.models.sc2_networks import SC2DuelingQNetV3
from utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 2, "Number of episodes to evaluate.")
flags.DEFINE_float("epsilon", 0.05, "Epsilon for policy.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent", 'random', ['dqn', 'random', 'keyboard'], "Algorithm.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("render", False, "Visualize feature map or not.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_boolean("flip_features", True, "Flip 2D features.")
flags.DEFINE_boolean("use_reward_shaping", False, "Enable reward shaping.")
flags.FLAGS(sys.argv)

np.random.seed(0)
random.seed(0)


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
    if FLAGS.use_reward_shaping:
        env = RewardShapingWrapperV2(env)
    env = ZergActionWrapper(env)
    print("----------------------------- Actions -----------------------------")
    env.print_actions()
    print("-------------------------------------------------------------------")
    env = ZergObservationWrapper(env, flip=FLAGS.flip_features)
    return env


def train():
    env = create_env()
    network = SC2DuelingQNetV3(
        resolution=env.observation_space.spaces[0].shape[1],
        n_channels=env.observation_space.spaces[0].shape[0],
        n_dims=env.observation_space.spaces[1].shape[0],
        n_out=env.action_space.n,
        batchnorm=FLAGS.use_batchnorm)

    if FLAGS.agent == 'dqn':
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
            double_dqn=True,
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
            step_id = 0
            while not done:
                action = agent.act(observation, eps=FLAGS.epsilon)
                observation, reward, done, _ = env.step(action)
                cum_return += reward
                print(step_id, action)
                assert step_id != 4 or action == 1
                assert step_id != 30 or action == 7
                assert step_id != 52 or action == 18
                assert step_id != 53 or action == 13
                assert step_id != 69 or action == 6
                assert step_id != 76 or action == 21
                assert step_id != 110 or action == 19
                assert step_id != 121 or action == 23
                assert step_id != 147 or action == 2
                assert step_id != 173 or action == 5
                assert step_id != 186 or action == 10
                assert step_id != 189 or action == 8
                assert step_id != 211 or action == 10
                assert step_id != 212 or action == 12
                assert step_id != 217 or action == 8
                assert step_id != 227 or action == 5
                assert step_id != 234 or action == 1
                assert step_id != 235 or action == 12
                assert step_id != 237 or action == 12
                assert step_id != 258 or action == 9
                assert step_id != 280 or action == 17
                assert step_id != 290 or action == 17
                assert step_id != 340 or action == 16
                assert step_id != 348 or action == 14
                if step_id >= 350:
                    break
                step_id += 1
            print("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f" %
                  (i + 1, FLAGS.num_episodes, cum_return / (i + 1),
                   ((cum_return / (i + 1)) + 1) / 2.0))
        print("All Tests Passed.")
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
