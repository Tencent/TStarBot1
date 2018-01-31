import numpy as np

import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

import random
import scipy.ndimage as ndimage

from pysc2.lib import actions
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib.actions import TYPES
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES
from pysc2.lib.features import FeatureType
from pysc2.env.sc2_env import SC2Env as PySC2Env


class SC2ScriptedEnv(gym.Env):

    def __init__(self,
                 map_name,
                 step_mul=8,
                 agent_race=None,
                 bot_race=None,
                 difficulty=None,
                 game_steps_per_episode=0,
                 resolution=64,
                 unittype_whitelist=None,
                 observation_filter=[]):
        self._resolution = resolution
        self._num_steps = 0
        self._sc2_env = PySC2Env(
            map_name=map_name,
            step_mul=step_mul,
            agent_race=agent_race,
            bot_race=bot_race,
            difficulty=difficulty,
            game_steps_per_episode=game_steps_per_episode,
            screen_size_px=(resolution, resolution),
            minimap_size_px=(resolution, resolution),
            visualize=False)

        self._unittype_map = None
        if unittype_whitelist:
            self._unittype_map = {v : i
                                  for i, v in enumerate(unittype_whitelist)}
        self._observation_filter = set(observation_filter) 

        self._last_raw_obs = None
        self._func_queue = []
        self._base_xy = None
        self._action_to_cmds = [self._cmds_idle,
                                self._cmds_train_scv,
                                self._cmds_build_barrack,
                                self._cmds_build_supply_depot,
                                self._cmds_train_marine,
                                self._cmds_move_camera_to_base,
                                self._cmds_all_defence,
                                self._cmds_all_attack]
    
    @property
    def action_spec(self):
        return (len(self._action_to_cmds),)

    @property
    def observation_spec(self):
        # TODO: to be refactor
        num_channels_screen, num_channels_minimap = self._get_observation_spec()
        return (num_channels_screen, num_channels_minimap, self._resolution)

    def _step(self, action):
        action = action[0]
        obs = self._last_obs
        self._locate_base(obs)
        step_taken = False
        for func in self._action_to_cmds[action](obs):
            func_id, func_args = func(obs)
            if not self._is_available(obs, func_id):
                break
            op =  actions.FunctionCall(func_id, func_args)
            obs = self._sc2_env.step([op])[0]
            step_taken = True
        if not step_taken:
            op =  actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
            obs = self._sc2_env.step([op])[0]
        self._last_obs = obs
        return self._transform_observation(obs)

    def _reset(self):
        timestep = self._sc2_env.reset()[0]
        self._last_obs = timestep
        return (self._transform_observation(timestep)[0],
                self._transform_observation(timestep)[3])

    def _close(self):
        self._sc2_env.close()

    def _transform_observation(self, timestep):
        obs_screen = self._transform_spatial_features(
            timestep.observation["screen"], SCREEN_FEATURES)
        obs_minimap = self._transform_spatial_features(
            timestep.observation["minimap"], MINIMAP_FEATURES)
        obs_player = self._transform_player_feature(
            timestep.observation["player"])
        obs = (obs_screen, obs_minimap, obs_player)
        done = timestep.last()
        return obs, timestep.reward, done, None

    def _transform_player_feature(self, obs):
        return np.log10(obs[1:].astype(np.float32) + 1)

    def _transform_spatial_features(self, obs, specs):
        features = []
        for ob, spec in zip(obs, specs):
            if spec.name in self._observation_filter:
                continue
            scale = spec.scale
            if spec.name == "unit_type" and self._unittype_map:
                ob = np.vectorize(lambda k: self._unittype_map.get(k, 0))(ob)
                scale = len(self._unittype_map)
            if spec.type == FeatureType.CATEGORICAL:
                features.append(np.eye(scale, dtype=np.float32)[ob][:, :, 1:])
            else:
                features.append(
                    np.expand_dims(np.log10(ob + 1, dtype=np.float32), axis=2))
        return np.transpose(np.concatenate(features, axis=2), (2, 0, 1))

    def _get_observation_spec(self):
        def get_spatial_channels(specs):
            num_channels = 0
            for spec in specs:
                if spec.name in self._observation_filter:
                    continue
                if spec.name == "unit_type" and self._unittype_map:
                    num_channels += len(self._unittype_map) - 1
                    continue
                if spec.type == FeatureType.CATEGORICAL:
                    num_channels += spec.scale - 1
                else:
                    num_channels += 1
            return num_channels
        num_channels_screen = get_spatial_channels(SCREEN_FEATURES)
        num_channels_minimap = get_spatial_channels(MINIMAP_FEATURES)
        return num_channels_screen, num_channels_minimap

    def _is_available(self, obs, func_id):
        #if not func_id in obs.observation["available_actions"]:
            #print(func_id, obs.observation["available_actions"])
        return func_id in obs.observation["available_actions"]

    def _cmds_train_scv(self, obs):
        functions = []
        if not self._is_command_center_selected(obs):
            functions += self._cmds_move_camera_to_base(obs)
            functions.append(self._select_command_center)
        functions.append(self._train_scv)
        return functions

    def _cmds_train_marine(self, obs):
        functions = []
        functions.append(self._move_camera_to_base)
        functions.append(self._select_barack)
        functions.append(self._train_marine)
        return functions

    def _cmds_build_barrack(self, obs):
        functions = []
        functions.append(self._select_worker)
        functions.append(self._move_camera_to_base)
        functions.append(self._build_barrack)
        return functions

    def _cmds_build_supply_depot(self, obs):
        functions = []
        functions.append(self._select_worker)
        functions.append(self._move_camera_to_base)
        functions.append(self._build_supply_depot)
        return functions

    def _cmds_move_camera_to_base(self, obs):
        functions = []
        functions.append(self._move_camera_to_base)
        return functions

    def _cmds_move_camera_to_enemy_base(self, obs):
        functions = []
        functions.append(self._move_camera_to_base)
        return functions

    def _cmds_all_defence(self, obs):
        functions = []
        functions.append(self._move_camera_to_base)
        functions.append(self._select_army)
        functions.append(self._attack_in_screen)
        return functions

    def _cmds_idle(self, obs):
        functions = []
        functions.append(lambda obs: (actions.FUNCTIONS.no_op.id, []))
        return functions

    def _cmds_all_attack(self, obs):
        functions = []
        if not self._has_enemy_in_screen(obs):
            functions.append(self._move_camera_to_enemy_base)
            functions.append(self._select_army)
            functions.append(self._attack_center_in_screen)
        else:
            functions.append(self._select_army)
            functions.append(self._attack_in_screen)
        return functions

    def _is_command_center_selected(self, obs):
        unittype = obs.observation["screen"][6]
        selected = obs.observation["screen"][5]
        return np.all(unittype[selected > 0] == 18)

    def _has_command_center_in_screen(self, obs):
        unittype = obs.observation["screen"][6]
        return np.any(unittype == 18)

    def _has_enemy_in_screen(self, obs):
        player_relative = obs.observation["screen"][5]
        return np.any(player_relative == 4)

    def _build_barrack(self, obs):
        xy = self._find_vacant_location(obs, 13)
        if xy is None:
            return -1, []
        function_id = actions.FUNCTIONS.Build_Barracks_screen.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _build_supply_depot(self, obs):
        xy = self._find_vacant_location(obs, 9)
        if xy is None:
            return -1, []
        function_id = actions.FUNCTIONS.Build_SupplyDepot_screen.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _find_vacant_location(self, obs, erosion_size):
        unittype = obs.observation["screen"][6]
        heightmap = obs.observation["screen"][0]
        vacant = (unittype == 0) & (heightmap == 255)
        vacant_erosed = ndimage.grey_erosion(
            vacant, size=(erosion_size, erosion_size))
        candidate_xy = np.transpose(np.nonzero(vacant_erosed)).tolist()
        if len(candidate_xy) == 0:
            return None
        xy = random.choice(candidate_xy)
        return xy

    def _find_enemy_location(self, obs):
        player_relative = obs.observation["screen"][5]
        candidate_xy = np.transpose(np.nonzero(player_relative == 5)).tolist()
        if len(candidate_xy) == 0:
            return None
        xy = random.choice(candidate_xy)
        return xy

    def _attack_in_screen(self, obs):
        xy = self._find_enemy_location(obs)
        if xy is None:
            return -1, []
        function_id = actions.FUNCTIONS.Attack_screen.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _attack_center_in_screen(self, obs):
        xy = [self._resolution / 2, self._resolution / 2]
        function_id = actions.FUNCTIONS.Attack_screen.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _attack_enemy_base(self, obs):
        xy = self._resolution - self._base_xy
        function_id = actions.FUNCTIONS.Attack_minimap.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _select_command_center(self, obs):
        unittype = obs.observation["screen"][6]
        candidate_xy = np.transpose(np.nonzero(unittype == 18)).tolist()
        if len(candidate_xy) == 0:
            return -1, []
        xy = random.choice(candidate_xy)
        function_id = actions.FUNCTIONS.select_point.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _select_barack(self, obs):
        unittype = obs.observation["screen"][6]
        barrack_map = ndimage.grey_erosion(unittype == 21, size=(7, 7))
        candidate_xy = np.transpose(np.nonzero(barrack_map)).tolist()
        if len(candidate_xy) == 0:
            return -1, []
        xy = random.choice(candidate_xy)
        function_id = actions.FUNCTIONS.select_point.id
        function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _select_worker(self, obs):
        function_id = actions.FUNCTIONS.select_idle_worker.id
        function_args = [[0]]
        if function_id not in obs.observation["available_actions"]:
            unittype = obs.observation["screen"][6]
            candidate_xy = np.transpose(np.nonzero(unittype == 45)).tolist()
            if len(candidate_xy) == 0:
                return -1, []
            xy = random.choice(candidate_xy)
            function_id = actions.FUNCTIONS.select_point.id
            function_args = [[0], xy[::-1]]
        return function_id, function_args

    def _select_army(self, obs):
        function_id = actions.FUNCTIONS.select_army.id
        function_args = [[0]]
        return function_id, function_args

    def _train_scv(self, obs):
        function_id = actions.FUNCTIONS.Train_SCV_quick.id
        function_args = [[0]]
        return function_id, function_args

    def _train_marine(self, obs):
        function_id = actions.FUNCTIONS.Train_Marine_quick.id
        function_args = [[0]]
        return function_id, function_args

    def _locate_base(self, obs):
        if self._base_xy is None:
            camera = obs.observation["minimap"][3]
            self._base_xy = np.median(
                np.transpose(np.nonzero(camera == 1)), 0).astype(int)

    def _move_camera_to_self_random(self, obs):
        player_relative = obs.observation["minimap"][5]
        xy = random.choice(
            np.transpose(np.nonzero(player_relative == 1)).tolist())
        function_id = actions.FUNCTIONS.move_camera.id
        function_args = [xy[::-1]]
        return actions.FunctionCall(function_id, function_args)

    def _move_camera_to_base(self, obs):
        function_id = actions.FUNCTIONS.move_camera.id
        if self._base_xy[0] < self._resolution / 2:
             offset = [2, 2]
             random_offset = np.random.randint(0, 1, size=2)
        elif self._base_xy[0] > self._resolution / 2:
             offset = [-2, 0]
             random_offset = np.random.randint(-1, 0, size=2)
        function_args = [self._base_xy[::-1] + offset]
        return function_id, function_args

    def _move_camera_to_enemy_base(self, obs):
        function_id = actions.FUNCTIONS.move_camera.id
        if self._base_xy[0] < self._resolution / 2:
            offset = [-2, 1]
            random_offset = np.random.randint(-1, 0, size=2)
        elif self._base_xy[0] > self._resolution / 2:
            offset = [0, 2]
            random_offset = np.random.randint(0, 1, size=2)
        function_args = [self._resolution - self._base_xy[::-1] + offset]
        return function_id, function_args

    def _move_camera_to_selected(self, obs):
        camera = obs.observation["minimap"][3]
        selected = obs.observation["screen"][5]
        current_xy = np.mean(
            np.transpose(np.nonzero(camera == 1)), 0).astype(int)
        target_xy = current_xy - (selected.mean() -
            [self._resolution / 2, self._resolution / 2]) / 5
        target_xy = target_xy.astype(np.int32)
        function_id = actions.FUNCTIONS.move_camera.id
        function_args = [target_xy[::-1]]
        return function_id, function_args
