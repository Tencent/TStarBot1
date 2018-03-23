import numpy as np
import gym
import pysc2.env.sc2_env
from pysc2.lib import actions

from envs.space import PySC2RawAction, PySC2RawObservation


class StarCraftIIEnv(gym.Env):

    def __init__(self,
                 map_name,
                 step_mul=8,
                 resolution=32,
                 agent_race=None,
                 bot_race=None,
                 difficulty=None,
                 game_steps_per_episode=0,
                 score_index=None,
                 visualize_feature_map=False):
        self._resolution = resolution
        self._sc2_env = pysc2.env.sc2_env.SC2Env(
            map_name=map_name,
            step_mul=step_mul,
            agent_race=agent_race,
            bot_race=bot_race,
            difficulty=difficulty,
            game_steps_per_episode=game_steps_per_episode,
            screen_size_px=(resolution, resolution),
            minimap_size_px=(resolution, resolution),
            visualize=visualize_feature_map,
            score_index=score_index)
        self.observation_space = PySC2RawObservation(
            self._sc2_env.observation_spec)
        self.action_space = PySC2RawAction(self._sc2_env.action_spec)
        self._reseted = False

    def _step(self, action):
        assert self._reseted
        assert self.action_space.contains(action, self._available_actions)
        op =  actions.FunctionCall(*action)
        timestep = self._sc2_env.step([op])[0]
        observation = timestep.observation
        self._available_actions = observation["available_actions"]
        reward = float(timestep.reward)
        done = timestep.last() 
        if done: self._reseted = False
        info = {"available_actions": self._available_actions}
        return (observation, reward, done, info)
        
    def _reset(self):
        timestep = self._sc2_env.reset()[0]
        observation = timestep.observation
        self._available_actions = observation["available_actions"]
        self._reseted = True
        info = {"available_actions": self._available_actions}
        return observation, info

    def _close(self):
        self._sc2_env.close()
