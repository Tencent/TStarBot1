import os
import traceback
from absl import app
from absl import flags

from envs.sc2_env import SC2Env
from envs.parallel_env import ParallelEnvWrapper 
from agents.a2c_agent import A2CAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 16, "Number of environments to run in parallel.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap.")
flags.DEFINE_float("rmsprop_lr", 3e-4, "Learning rate for RMSProp.")
flags.DEFINE_float("rmsprop_eps", 1e-5, "Epsilon for RMSProp.")
flags.DEFINE_integer("rollout_num_steps", 5, "Rollout steps for A2C.")
flags.DEFINE_boolean("use_gpu", True, "Use gpu or not.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_model_freq", "10000", "Model saving frequency.")
flags.mark_flag_as_required("map")


def train():
    if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
        os.mkdir(FLAGS.save_model_dir)
    envs = ParallelEnvWrapper([lambda: SC2Env(
        map_name=FLAGS.map,
        step_mul=FLAGS.step_mul,
        screen_size_px=(FLAGS.resolution, FLAGS.resolution),
        action_filter=[],
        observation_filter=[]) for _ in range(FLAGS.n_envs)])
    agent = A2CAgent(
        dims=FLAGS.resolution,
        observation_spec=envs.observation_spec,
        action_spec=envs.action_spec,
        rmsprop_lr=FLAGS.rmsprop_lr,
        rmsprop_eps=FLAGS.rmsprop_eps,
        rollout_num_steps=FLAGS.rollout_num_steps,
        use_gpu=FLAGS.use_gpu,
        init_model_path=FLAGS.init_model_path,
        save_model_dir=FLAGS.save_model_dir,
        save_model_freq=FLAGS.save_model_freq)
    try:
        agent.train(envs)
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()
    envs.close()


def main(argv):
    train()


if __name__ == '__main__':
    app.run(main)
