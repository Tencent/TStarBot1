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
flags.DEFINE_string("init_model_path", None, "Filepath to load initial model.")
flags.DEFINE_enum("agent_race", 'T', ['P', 'Z', 'R', 'T'], "Agent's race.")
flags.DEFINE_enum("bot_race", 'T', ['P', 'Z', 'R', 'T'], "Bot's race.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', 'A', '3', '2', '5', '4', '7', '6', '9', '8'],
                  "Bot's strength.")
flags.mark_flag_as_required("map")


def train():
    env = SC2Env(map_name=FLAGS.map,
                 step_mul=FLAGS.step_mul,
                 agent_race=FLAGS.agent_race,
                 bot_race=FLAGS.bot_race,
                 difficulty=FLAGS.difficulty,
                 screen_size_px=(FLAGS.resolution, FLAGS.resolution),
                 action_filter=[],
                 observation_filter=[])

    agent = SLAgent(
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        use_gpu=FLAGS.use_gpu,
        init_model_path=FLAGS.init_model_path)
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
