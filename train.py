import sys
import torch
import os
import traceback
from absl import app
from absl import flags
import random

from envs.sc2_env import StarCraftIIEnv
from envs.actions.zerg_action_wrappers import ZergActionWrapper
from envs.observations.zerg_observation_wrappers import ZergObservationWrapper
from envs.rewards.reward_wrappers import RewardShapingWrapperV2
from agents.fast_dqn_agent import FastDQNAgent
from agents.models.sc2_networks import SC2DuelingQNetV3
from utils.utils import print_arguments


FLAGS = flags.FLAGS
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_integer("num_actor_workers", 16, "Game steps per agent step.")
flags.DEFINE_string("difficulty", '4', "Bot's strengths.")
flags.DEFINE_integer("memory_size", 250000, "Experience replay size.")
flags.DEFINE_integer("init_memory_size", 100000, "Experience replay init size.")
flags.DEFINE_enum("eps_method", 'linear', ['exponential', 'linear'],
                  "Epsilon decay methods.")
flags.DEFINE_float("eps_start", 1.0, "Max greedy epsilon for exploration.")
flags.DEFINE_float("eps_end", 0.1, "Min greedy epsilon for exploration.")
flags.DEFINE_integer("eps_decay", 5000000, "Greedy epsilon decay step.")
flags.DEFINE_integer("eps_decay2", 30000000, "Greedy epsilon decay step.")
flags.DEFINE_enum("optimizer_type", 'adam', ['rmsprop', 'adam', 'sgd'],
                  "Optimizer.")
flags.DEFINE_float("learning_rate", 3e-7, "Learning rate.")
flags.DEFINE_float("momentum", 0.9, "Momentum.")
flags.DEFINE_float("adam_eps", 1e-7, "Adam optimizer's epsilon.")
flags.DEFINE_float("gradient_clipping", 10.0, "Gradient clipping threshold.")
flags.DEFINE_float("frame_step_ratio", 1.0, "Actor frames per train step.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_float("discount", 0.995, "Discount.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_enum("loss_type", 'mse', ['mse', 'smooth_l1'], "Loss type.")
flags.DEFINE_integer("target_update_freq", 10000, "Target net update frequency.")
flags.DEFINE_integer("save_model_freq", 500000, "Model saving frequency.")
flags.DEFINE_integer("print_freq", 5000, "Print train cost frequency.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_boolean("flip_features", True, "Flip 2D features.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_boolean("use_reward_shaping", False, "Enable reward shaping.")
flags.FLAGS(sys.argv)


def create_env():
    difficulty = random.choice(FLAGS.difficulty.split(','))
    env = StarCraftIIEnv(
        map_name='AbyssalReef',
        step_mul=FLAGS.step_mul,
        resolution=32,
        agent_race='Z',
        bot_race='Z',
        difficulty=difficulty,
        disable_fog=FLAGS.disable_fog,
        game_steps_per_episode=0,
        visualize_feature_map=False,
        score_index=None)
    if FLAGS.use_reward_shaping:
        env = RewardShapingWrapperV2(env)
    env = ZergActionWrapper(env)
    env = ZergObservationWrapper(env, flip=FLAGS.flip_features)
    return env, difficulty


def train():
    if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
        os.makedirs(FLAGS.save_model_dir)

    env, _ = create_env()
    network = SC2DuelingQNetV3(
        resolution=env.observation_space.spaces[0].shape[1],
        n_channels=env.observation_space.spaces[0].shape[0],
        n_dims=env.observation_space.spaces[1].shape[0],
        n_out=env.action_space.n,
        batchnorm=FLAGS.use_batchnorm)

    agent = FastDQNAgent(
        observation_space=env.observation_space,
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
        allow_eval_mode=True,
        loss_type=FLAGS.loss_type,
        init_model_path=FLAGS.init_model_path,
        save_model_dir=FLAGS.save_model_dir,
        save_model_freq=FLAGS.save_model_freq,
        print_freq=FLAGS.print_freq)

    try:
        agent.learn(create_env, FLAGS.num_actor_workers)
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
