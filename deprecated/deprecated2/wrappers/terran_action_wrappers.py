import numpy as np
import random
import scipy.ndimage as ndimage

from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES
from pysc2.lib.features import MINIMAP_FEATURES

import gym
from gym.spaces.discrete import Discrete

from envs.space import PySC2ActionSpace
from envs.space import PySC2ObservationSpace

UNIT_TYPE_BACKGROUND = 0
UNIT_TYPE_COMMAND_CENTER = 18
UNIT_TYPE_WORKER = 45
UNIT_TYPE_BARRACK = 21
UNIT_TYPE_MINERAL = 483

PLAYER_RELATIVE_ENEMY = 4
PLAYERINFO_IDLE_WORKER_COUNT = 7

NOT_QUEUED = 0
QUEUED = 1

SELECT = 0
SELECT_ADD = 1

SELECT_IDLE_WORKER_ANY = 0
SELECT_IDLE_WORKER_ADDANY = 1
SELECT_IDLE_WORKER_ALL = 2
SELECT_IDLE_WORKER_ADDALL = 3

SELECT_POINT_SELECT = 0
SELECT_POINT_TOGGLE = 1


def has_command_center_screen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    return np.any(unit_type == UNIT_TYPE_COMMAND_CENTER)


def has_enemy_screen(observation):
    player_relative = observation["screen"][
        SCREEN_FEATURES.player_relative.index]
    return np.any(player_relative == PLAYER_RELATIVE_ENEMY)


def is_command_center_selected(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    selected = observation["screen"][SCREEN_FEATURES.selected.index]
    return np.all(unit_type[selected > 0] == UNIT_TYPE_COMMAND_CENTER)


def find_vacant_location(observation, erosion_size):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    height_map = observation["screen"][SCREEN_FEATURES.height_map.index]
    vacant = (unit_type == UNIT_TYPE_BACKGROUND) & (height_map == 255)
    vacant_erosed = ndimage.grey_erosion(
        vacant, size=(erosion_size, erosion_size))
    candidate_xy = np.transpose(np.nonzero(vacant_erosed)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_enemy_screen(observation):
    player_relative = observation["screen"][
        SCREEN_FEATURES.player_relative.index]
    candidate_xy = np.transpose(
        np.nonzero(player_relative == PLAYER_RELATIVE_ENEMY)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_enemy_minimap(observation):
    player_relative = observation["minimap"][
        MINIMAP_FEATURES.player_relative.index]
    candidate_xy = np.transpose(
        np.nonzero(player_relative == PLAYER_RELATIVE_ENEMY)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_minerals_screen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    minerals = ndimage.grey_erosion(unit_type == UNIT_TYPE_MINERAL,
                                    size=(3, 3))
    candidate_xy = np.transpose(np.nonzero(minerals)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def macro_train_scv(observation):
    micros = []
    if not is_command_center_selected(observation):
        micros.extend(macro_move_camera_to_self_base(observation))
        micros.append(micro_select_command_center)
    micros.append(micro_train_scv)
    return micros


def macro_train_marine(observation):
    micros = []
    micros.append(micro_move_camera_to_self_base)
    micros.append(micro_select_barack)
    micros.extend([micro_train_marine] * 5)
    return micros


def macro_build_barrack(observation):
    micros = []
    micros.append(micro_select_any_worker)
    micros.append(micro_move_camera_to_self_base)
    micros.append(micro_build_barrack)
    return micros


def macro_build_supply_depot(observation):
    micros = []
    micros.append(micro_select_any_worker)
    micros.append(micro_move_camera_to_self_base)
    micros.append(micro_build_supply_depot)
    return micros


def macro_move_camera_to_self_base(observation):
    micros = [micro_move_camera_to_self_base]
    return micros


def macro_move_camera_to_enemy_base(observation):
    micros = [micro_move_camera_to_enemy_base]
    return micros


def macro_all_defence(observation):
    micros = []
    micros.append(micro_move_camera_to_self_base)
    micros.append(micro_select_all_armies)
    micros.append(micro_attack_any_screen)
    return micros


def macro_do_nothing(observation):
    micros = [micro_do_nothing]
    return micros


def macro_all_attack_enemy_base(observation):
    micros = []
    if not has_enemy_screen(observation):
        micros.append(micro_move_camera_to_enemy_base)
        micros.append(micro_select_all_armies)
        micros.append(micro_attack_center_screen)
    else:
        micros.append(micro_select_all_armies)
        micros.append(micro_attack_any_screen)
    return micros


def macro_all_attack_any_enemy(observation):
    micros = []
    micros.append(micro_move_camera_to_any_enemy)
    micros.append(micro_select_all_armies)
    micros.append(micro_attack_any_screen)
    return micros


def macro_all_idle_workers_collect_minerals(observation):
    micros = []
    micros.append(micro_move_camera_to_self_base)
    micros.append(micro_select_all_idle_workers)
    micros.append(micro_go_to_minerals)
    return micros


def micro_train_scv(observation):
    function_id = actions.FUNCTIONS.Train_SCV_quick.id
    function_args = [[0]]
    return function_id, function_args


def micro_build_barrack(observation):
    xy = find_vacant_location(observation, 7)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Build_Barracks_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_build_supply_depot(observation):
    xy = find_vacant_location(observation, 5)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Build_SupplyDepot_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_attack_any_screen(observation):
    xy = find_enemy_screen(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Attack_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_attack_center_screen(observation):
    resolution = observation["screen"].shape[-1]
    xy = [resolution / 2, resolution / 2]
    function_id = actions.FUNCTIONS.Attack_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_attack_enemy_base(observation):
    resolution = observation["screen"].shape[-1]
    xy = resolution - observation["base_xy"]
    function_id = actions.FUNCTIONS.Attack_minimap.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_go_to_minerals(observation):
    xy = find_minerals_screen(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Smart_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_select_command_center(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    candidate_xy = np.transpose(
        np.nonzero(unit_type == UNIT_TYPE_COMMAND_CENTER)).tolist()
    if len(candidate_xy) == 0: return None
    xy = random.choice(candidate_xy)
    function_id = actions.FUNCTIONS.select_point.id
    function_args = [[SELECT_POINT_SELECT], xy[::-1]]
    return function_id, function_args


def micro_select_barack(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    barrack_map = ndimage.grey_erosion(unit_type == UNIT_TYPE_BARRACK,
                                       size=(3, 3))
    candidate_xy = np.transpose(np.nonzero(barrack_map)).tolist()
    if len(candidate_xy) == 0: return None
    xy = random.choice(candidate_xy)
    function_id = actions.FUNCTIONS.select_point.id
    function_args = [[SELECT_POINT_SELECT], xy[::-1]]
    return function_id, function_args


def micro_select_any_worker(observation):
    if observation["player"][PLAYERINFO_IDLE_WORKER_COUNT] > 0:
        function_id = actions.FUNCTIONS.select_idle_worker.id
        function_args = [[SELECT_IDLE_WORKER_ANY]]
    else:
        unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
        candidate_xy = np.transpose(
            np.nonzero(unit_type == UNIT_TYPE_WORKER)).tolist()
        if len(candidate_xy) == 0: return None
        xy = random.choice(candidate_xy)
        function_id = actions.FUNCTIONS.select_point.id
        function_args = [[SELECT_POINT_SELECT], xy[::-1]]
    return function_id, function_args


def micro_select_all_idle_workers(observation):
    if observation["player"][PLAYERINFO_IDLE_WORKER_COUNT] == 0:
        return None
    function_id = actions.FUNCTIONS.select_idle_worker.id
    function_args = [[SELECT_IDLE_WORKER_ALL]]
    return function_id, function_args


def micro_select_all_armies(observation):
    function_id = actions.FUNCTIONS.select_army.id
    function_args = [[SELECT]]
    return function_id, function_args


def micro_train_marine(observation):
    function_id = actions.FUNCTIONS.Train_Marine_quick.id
    function_args = [[NOT_QUEUED]]
    return function_id, function_args


def micro_move_camera_to_any_enemy(observation):
    xy = find_enemy_minimap(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.move_camera.id
    function_args = [xy[::-1]]
    return actions.FunctionCall(function_id, function_args)


def micro_move_camera_to_self_base(observation):
    function_id = actions.FUNCTIONS.move_camera.id
    resolution = observation["screen"].shape[-1]
    if observation["base_xy"][0] < resolution / 2:
         offset = [2, 2]
    else:
         offset = [0, 0]
    function_args = [observation["base_xy"][::-1] + offset]
    return function_id, function_args


def micro_move_camera_to_enemy_base(observation):
    function_id = actions.FUNCTIONS.move_camera.id
    resolution = observation["screen"].shape[-1]
    if observation["base_xy"][0] < resolution / 2:
        offset = [0, 0]
    else:
        offset = [0, 1]
    function_args = [resolution - observation["base_xy"][::-1] + offset]
    return function_id, function_args


def micro_do_nothing(observation):
    function_id = actions.FUNCTIONS.no_op.id
    function_args = []
    return function_id, function_args


def locate_camera_minimap(observation):
    camera = observation["minimap"][MINIMAP_FEATURES.camera.index]
    return np.median(np.transpose(np.nonzero(camera == 1)), 0).astype(int)


class TerranActionWrapperV0(gym.Wrapper):

    def __init__(self, env):
        super(TerranActionWrapperV0, self).__init__(env)
        assert isinstance(env.action_space, PySC2ActionSpace)
        assert isinstance(env.observation_space, PySC2ObservationSpace)
        shape_screen = self.env.observation_space.space_attr["screen"][1:]
        shape_minimap = self.env.observation_space.space_attr["minimap"][1:]
        assert shape_screen == (32, 32) and shape_minimap == (32, 32)
        self._macro_actions = [macro_do_nothing,
                               macro_build_supply_depot,
                               macro_build_barrack,
                               macro_train_scv,
                               macro_train_marine,
                               macro_all_defence,
                               macro_all_attack_enemy_base,
                               macro_all_attack_any_enemy,
                               macro_all_idle_workers_collect_minerals,
                               macro_move_camera_to_self_base]
        self.action_space = Discrete(len(self._macro_actions))

    def step(self, action):
        macro_action = self._macro_actions[action]
        observation = self._last_observation
        observation["base_xy"] = self._base_xy
        reward_cum = 0
        acted = False
        for micro_action in macro_action(observation):
            action = micro_action(observation)
            if (action is None or
                self.env.action_space.contains(
                    action, observation["available_actions"]) == False):
                break
            observation, reward, done, info = self.env.step(action)
            observation["base_xy"] = self._base_xy
            acted = True
            reward_cum += reward
            if done:
                return observation, reward_cum, done, info
        if not acted:
            action = micro_do_nothing(observation)
            observation, reward, done, info = self.env.step(action)
            observation["base_xy"] = self._base_xy
            reward_cum += reward
        self._last_observation = observation 
        return observation, reward_cum, done, info
            
    def reset(self):
        observation = self.env.reset()
        self._base_xy = locate_camera_minimap(observation)
        self._last_observation = observation
        return observation
