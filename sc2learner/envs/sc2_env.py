from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import pysc2.env.sc2_env

from sc2learner.envs.spaces.pysc2_raw import PySC2RawAction
from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation


class StarCraftIIEnv(gym.Env):

  def __init__(self,
               map_name,
               step_mul=8,
               resolution=32,
               disable_fog=False,
               agent_race=None,
               bot_race=None,
               difficulty=None,
               game_steps_per_episode=0,
               score_index=None,
               visualize_feature_map=False,
               random_seed=None):
    self._resolution = resolution
    self._sc2_env = pysc2.env.sc2_env.SC2Env(
        map_name=map_name,
        step_mul=step_mul,
        agent_race=agent_race,
        bot_race=bot_race,
        disable_fog=disable_fog,
        difficulty=difficulty,
        game_steps_per_episode=game_steps_per_episode,
        screen_size_px=(resolution, resolution),
        minimap_size_px=(resolution, resolution),
        visualize=visualize_feature_map,
        score_index=score_index,
        random_seed=random_seed)
    self.observation_space = PySC2RawObservation(self._sc2_env.observation_spec)
    self.action_space = None
    self._difficulty = difficulty
    self._reseted = False

  def _step(self, actions):
    assert self._reseted
    timestep = self._sc2_env.step([actions])[0]
    observation = timestep.observation
    reward = float(timestep.reward)
    done = timestep.last()
    if done:
      self._reseted = False
      print("Episode Done. Difficulty: %s Outcome %f" %
            (self._difficulty, reward))
    info = {}
    return (observation, reward, done, info)

  def _reset(self):
    timestep = self._sc2_env.reset()[0]
    observation = timestep.observation
    self._reseted = True
    return observation

  def _close(self):
    self._sc2_env.close()
