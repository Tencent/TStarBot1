import torch,cv2
from absl import app
from absl import flags

from envs.sc2env import SC2MiniEnv
from envs.parallel_env import ParallelEnvWrapper 
from agents.a2c_agent import A2CAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 2, "Number of environments to run in parallel.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap.")
flags.DEFINE_integer("select_army_freq", 5, "Frequency for re-selecting army.")
flags.DEFINE_float("rmsprop_lr", 3e-4, "Learning rate for RMSProp.")
flags.DEFINE_float("rmsprop_eps", 1e-5, "Epsilon for RMSProp.")
flags.DEFINE_integer("rollout_num_steps", 5, "Rollout steps for A2C.")
flags.mark_flag_as_required("map")


def train():
    envs = ParallelEnvWrapper([lambda: SC2MiniEnv(
        map_name=FLAGS.map,
        step_mul=FLAGS.step_mul,
        screen_size_px=(FLAGS.resolution, FLAGS.resolution),
        select_army_freq=FLAGS.select_army_freq) for _ in range(FLAGS.n_envs)])
    agent = A2CAgent(
        dims=FLAGS.resolution,
        rmsprop_lr=FLAGS.rmsprop_lr,
        rmsprop_eps=FLAGS.rmsprop_eps,
        rollout_num_steps=FLAGS.rollout_num_steps)
    agent.train(envs)


def main(argv):
    train()


if __name__ == '__main__':
    app.run(main)
