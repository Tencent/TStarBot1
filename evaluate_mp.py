import sys
import torch
import os
import traceback
from absl import app
from absl import flags
import multiprocessing
import time

from envs.sc2_env_unit_control import StarCraftIIEnv
from wrappers.zerg_action_unit_control_wrappers import ZergActionWrapper
from wrappers.zerg_observation_wrappers import ZergObservationWrapper
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.fast_dqn_agent import FastDQNAgent
from models.sc2_networks import SC2DuelingQNetV3
from utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_parallels", 4, "Parallel number.")
flags.DEFINE_integer("num_episodes", 50, "Number of episodes to evaluate.")
flags.DEFINE_float("epsilon", 0.05, "Epsilon for policy.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '2',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent", 'dqn', ['dqn', 'random', 'keyboard'], "Algorithm.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("render", True, "Visualize feature map or not.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_boolean("flip_features", True, "Flip 2D features.")
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
    env = ZergObservationWrapper(env, flip=FLAGS.flip_features)
    return env


def train(pid):
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
            if (i + 1) % 5 == 0:
                env.close()
                env = create_env()
            observation = env.reset()
            done = False
            while not done:
                action = agent.act(observation, eps=FLAGS.epsilon)
                observation, reward, done, _ = env.step(action)
                cum_return += reward
            print("Process: %d Episode: %d Outcome: %f" % (pid, i, reward))
            print("Process: %d Evaluated %d/%d Episodes Avg Return %f "
                  "Avg Winning Rate %f" %
                  (pid, i + 1, FLAGS.num_episodes, cum_return / (i + 1),
                   ((cum_return / (i + 1)) + 1) / 2.0))
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()
    env.close()


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
