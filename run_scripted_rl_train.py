import os
import traceback
from absl import app
from absl import flags

from envs.sc2_scripted_env import SC2ScriptedEnv
from envs.parallel_env import ParallelEnvWrapper 
from agents.a2c_scripted_agent import A2CScriptedAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("map", 'AbyssalReef', "Name of a map to use.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 32, "Number of environments to run in parallel.")
flags.DEFINE_integer("resolution", 64, "Resolution for screen and minimap.")
flags.DEFINE_float("rmsprop_lr", 1e-5, "Learning rate for RMSProp.")
flags.DEFINE_float("rmsprop_eps", 1e-5, "Epsilon for RMSProp.")
flags.DEFINE_integer("rollout_num_steps", 20, "Rollout steps for A2C.")
flags.DEFINE_boolean("use_gpu", True, "Use gpu or not.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_string("save_model_dir", "./checkpoints/", "Dir to save models to")
flags.DEFINE_integer("save_model_freq", "100", "Model saving frequency.")
flags.DEFINE_enum("agent_race", 'T', ['P', 'Z', 'R', 'T'], "Agent's race.")
flags.DEFINE_enum("bot_race", 'T', ['P', 'Z', 'R', 'T'], "Bot's race.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', 'A', '3', '2', '5', '4', '7', '6', '9', '8'],
                  "Bot's strength.")
flags.DEFINE_string("observation_filter", "", "Observation field to ignore.")

unittype_whitelist = [0, 5, 6, 11, 18, 19, 20, 21, 22, 23,
                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                      34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                      44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                      54, 55, 56, 57, 58, 130, 132, 134, 146, 147,
                      149, 268, 341, 342, 343, 365, 472, 473, 474, 483,
                      484, 490, 498, 500, 561, 609, 638, 639, 640, 641,
                      662, 665, 666, 689, 691, 692, 734, 830, 880, 1879,
                      1883]

def train():
    if FLAGS.save_model_dir and not os.path.exists(FLAGS.save_model_dir):
        os.mkdir(FLAGS.save_model_dir)
    envs = ParallelEnvWrapper([lambda: SC2ScriptedEnv(
        map_name=FLAGS.map,
        step_mul=FLAGS.step_mul,
        agent_race=FLAGS.agent_race,
        bot_race=FLAGS.bot_race,
        difficulty=FLAGS.difficulty,
        resolution=FLAGS.resolution,
        unittype_whitelist=unittype_whitelist,
        observation_filter=FLAGS.observation_filter.split(","))
        for _ in range(FLAGS.n_envs)])
    agent = A2CScriptedAgent(
        observation_spec=envs.observation_spec,
        action_spec=envs.action_spec,
        rmsprop_lr=FLAGS.rmsprop_lr,
        rmsprop_eps=FLAGS.rmsprop_eps,
        rollout_num_steps=FLAGS.rollout_num_steps,
        use_gpu=FLAGS.use_gpu,
        init_model_path=FLAGS.init_model_path,
        save_model_dir=FLAGS.save_model_dir,
        save_model_freq=FLAGS.save_model_freq,
        enable_batchnorm=FLAGS.use_batchnorm)
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
