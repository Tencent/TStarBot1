import sys
from absl import flags
from absl import app

import pysc2.env.sc2_env
from pysc2.lib.actions import FUNCTIONS

from envs.sc2_env import StarCraftIIEnv
from wrappers.terran_action_wrappers import TerranActionWrapperV0
from wrappers.sc2_observation_wrappers import SC2ObservationWrapper

flags.FLAGS(sys.argv)


def test():
    env = StarCraftIIEnv(
        map_name='AbyssalReef',
        step_mul=8,
        resolution = 32,
        agent_race=None,
        bot_race=None,
        difficulty=None,
        game_steps_per_episode=0,
        visualize_feature_map=False,
        score_index=None)
    env = TerranActionWrapperV0(env)
    env = SC2ObservationWrapper(env)
    print("Action Space: %s" % env.action_space)
    print("Observation Space :%s" % env.observation_space)

    obs = env.reset()
    screen, minimap, player = obs
    assert screen.shape == env.observation_space.spaces[0].shape
    assert minimap.shape == env.observation_space.spaces[1].shape
    assert player.shape == env.observation_space.spaces[2].shape
    print("Reset Test Done.")

    for action in range(env.action_space.n):
        obs, reward, done, info = env.step(action)
        screen, minimap, player = obs
        assert screen.shape == env.observation_space.spaces[0].shape
        assert minimap.shape == env.observation_space.spaces[1].shape
        assert player.shape == env.observation_space.spaces[2].shape
        print("Action=%d Test Done." % action)

    env.close()
    print("All Test Done.")


def main(argv):
    test()


if __name__ == '__main__':
    app.run(main)
