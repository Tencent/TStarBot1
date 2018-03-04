import sys
from absl import flags
from absl import app

import pysc2.env.sc2_env
from pysc2.lib.actions import FUNCTIONS

from envs.space import PySC2ActionSpace, PySC2ObservationSpace
from envs.sc2_env import StarCraftIIEnv
from wrappers.terran_action_wrappers import TerranActionWrapperV0
from wrappers.sc2_observation_wrappers import SC2ObservationWrapper

flags.FLAGS(sys.argv)


def test():
    sc2_env = StarCraftIIEnv(
        map_name='AbyssalReef',
        step_mul=8,
        resolution = 32,
        agent_race=None,
        bot_race=None,
        difficulty=None,
        game_steps_per_episode=0,
        visualize_feature_map=False,
        score_index=None)
    sc2_env = TerranActionWrapperV0(sc2_env)
    sc2_env = SC2ObservationWrapper(sc2_env)
    print("Action Space: %s" % sc2_env.action_space)
    print("Observation Space :%s" % sc2_env.observation_space)

    obs = sc2_env.reset()
    screen, minimap, player = obs
    assert screen.shape == sc2_env.observation_space.spaces[0].shape
    assert minimap.shape == sc2_env.observation_space.spaces[1].shape
    assert player.shape == sc2_env.observation_space.spaces[2].shape
    print("Reset Test Done.")

    for action in range(sc2_env.action_space.n):
        obs, reward, done, info = sc2_env.step(action)
        screen, minimap, player = obs
        assert screen.shape == sc2_env.observation_space.spaces[0].shape
        assert minimap.shape == sc2_env.observation_space.spaces[1].shape
        assert player.shape == sc2_env.observation_space.spaces[2].shape
        print("Action=%d Test Done." % action)

    sc2_env.close()
    print("All Test Done.")


def main(argv):
    test()


if __name__ == '__main__':
    app.run(main)
