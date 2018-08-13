from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from pysc2.env import sc2_env

from sc2learner.envs.spaces.pysc2_raw import PySC2RawAction
from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation
from sc2learner.utils.utils import tprint


DIFFICULTIES= {
    "1": sc2_env.Difficulty.very_easy,
    "2": sc2_env.Difficulty.easy,
    "3": sc2_env.Difficulty.medium,
    "4": sc2_env.Difficulty.medium_hard,
    "5": sc2_env.Difficulty.hard,
    "6": sc2_env.Difficulty.hard,
    "7": sc2_env.Difficulty.very_hard,
    "8": sc2_env.Difficulty.cheat_vision,
    "9": sc2_env.Difficulty.cheat_money,
    "A": sc2_env.Difficulty.cheat_insane,
}


class SC2RawEnv(gym.Env):

  def __init__(self,
               map_name,
               step_mul=8,
               resolution=32,
               disable_fog=False,
               agent_race='random',
               bot_race='random',
               difficulty='1',
               game_steps_per_episode=None,
               score_index=None,
               random_seed=None):
    players=[sc2_env.Agent(sc2_env.Race[agent_race]),
             sc2_env.Bot(sc2_env.Race[bot_race], DIFFICULTIES[difficulty])]
    agent_interface_format=sc2_env.parse_agent_interface_format(
        feature_screen=resolution, feature_minimap=resolution)
    self._sc2_env = sc2_env.SC2Env(
        map_name=map_name,
        step_mul=step_mul,
        players=players,
        agent_interface_format=agent_interface_format,
        disable_fog=disable_fog,
        game_steps_per_episode=game_steps_per_episode,
        visualize=False,
        score_index=score_index,
        random_seed=random_seed)
    self.observation_space = PySC2RawObservation(self._sc2_env.observation_spec)
    self.action_space = PySC2RawAction()
    self._difficulty = difficulty
    self._reseted = False

  def step(self, actions):
    assert self._reseted
    timestep = self._sc2_env.step([actions])[0]
    observation = timestep.observation
    reward = float(timestep.reward)
    done = timestep.last()
    if done:
      self._reseted = False
      tprint("Episode Done. Difficulty: %s Outcome %f" %
             (self._difficulty, reward))
    info = {}
    return (observation, reward, done, info)

  def reset(self):
    timestep = self._sc2_env.reset()[0]
    observation = timestep.observation
    self._reseted = True
    return observation

  def close(self):
    self._sc2_env.close()
