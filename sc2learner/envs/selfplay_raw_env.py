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


class SC2SelfplayRawEnv(gym.Env):

  def __init__(self,
               map_name,
               step_mul=8,
               resolution=32,
               disable_fog=False,
               agent_race='random',
               opponent_race='random',
               game_steps_per_episode=None,
               score_index=None,
               random_seed=None):
    players=[sc2_env.Agent(sc2_env.Race[agent_race]),
             sc2_env.Agent(sc2_env.Race[opponent_race])]
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
    self._reseted = False

  def step(self, actions):
    assert self._reseted
    assert len(actions) == 2
    timesteps = self._sc2_env.step(actions)
    observation = [timesteps[0].observation, timesteps[1].observation]
    reward = float(timesteps[0].reward)
    done = timesteps[0].last()
    if done:
      self._reseted = False
      tprint("Episode Done. Outcome %f" % reward)
    info = {}
    return (observation, reward, done, info)

  def reset(self, **kwargs):
    timesteps = self._sc2_env.reset()
    observation = [timesteps[0].observation, timesteps[1].observation]
    self._reseted = True
    return observation

  def close(self):
    self._sc2_env.close()
