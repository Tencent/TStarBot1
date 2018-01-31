import os
import traceback
from absl import app
from absl import flags

from envs.sc2_env import SC2Env
from agents.sl_agent import SLAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("resolution", 64, "Resolution for screen and minimap.")
flags.DEFINE_boolean("use_gpu", True, "Use gpu or not.")
flags.DEFINE_boolean("use_batchnorm", False, "Use batchnorm or not.")
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent_race", 'T', ['P', 'Z', 'R', 'T'], "Agent's race.")
flags.DEFINE_enum("bot_race", 'T', ['P', 'Z', 'R', 'T'], "Bot's race.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', 'A', '3', '2', '5', '4', '7', '6', '9', '8'],
                  "Bot's strength.")
flags.DEFINE_string("observation_filter", "", "Observation field to ignore.")
flags.mark_flag_as_required("map")

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
    env = SC2Env(map_name=FLAGS.map,
                 step_mul=FLAGS.step_mul,
                 agent_race=FLAGS.agent_race,
                 bot_race=FLAGS.bot_race,
                 difficulty=FLAGS.difficulty,
                 screen_size_px=(FLAGS.resolution, FLAGS.resolution),
                 action_filter=[],
                 unittype_whitelist=unittype_whitelist,
                 observation_filter=FLAGS.observation_filter.split(","))

    agent = SLAgent(
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        use_gpu=FLAGS.use_gpu,
        init_model_path=FLAGS.init_model_path,
        enable_batchnorm=FLAGS.use_batchnorm)
    try:
        ob, info = env.reset()
        done = False
        while not done: 
            action = agent.step(ob, info)
            ob, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()


def main(argv):
    train()


if __name__ == '__main__':
    app.run(main)
