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
UNIT_TYPE_HATCHERY = 86
UNIT_TYPE_WORKER = 104
UNIT_TYPE_SPAWNING_POOL = 89
UNIT_TYPE_ROACH_WARREN = 97
UNIT_TYPE_VESPENE = 342
UNIT_TYPE_MINERAL = 483
UNIT_TYPE_EXTRACTOR = 88
UNIT_TYPE_QUEEN = 126

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


def has_spawning_pool_screen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    return np.any(unit_type == UNIT_TYPE_SPAWNING_POOL)


def has_roach_warren_screen(observation):
    return False
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    return np.any(unit_type == UNIT_TYPE_SPAWNING_POOL)


def has_enemy_screen(observation):
    player_relative = observation["screen"][
        SCREEN_FEATURES.player_relative.index]
    return np.any(player_relative == PLAYER_RELATIVE_ENEMY)


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


def find_vacant_creep_location(observation, erosion_size):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    creep = observation["screen"][SCREEN_FEATURES.creep.index]
    vacant = (unit_type == UNIT_TYPE_BACKGROUND) & (creep == 1)
    vacant_erosed = ndimage.grey_erosion(
        vacant, size=(erosion_size, erosion_size))
    candidate_xy = np.transpose(np.nonzero(vacant_erosed)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_location_far_from_hatchery(observation, radius):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    hatchery = unit_type == UNIT_TYPE_HATCHERY
    hatchery_dilated = ndimage.grey_dilation(hatchery, size=(radius, radius))
    candidate_xy = np.transpose(np.nonzero(1 - hatchery_dilated)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_minerals_screen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    minerals = unit_type == UNIT_TYPE_MINERAL
    candidate_xy = np.transpose(np.nonzero(minerals)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_vespene_screen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    vespene = ndimage.grey_erosion(unit_type == UNIT_TYPE_VESPENE, size=(2, 2))
    vespene = ndimage.binary_erosion(vespene)
    candidate_xy = np.transpose(np.nonzero(vespene)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)


def find_extractor_screen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    extractor = ndimage.grey_erosion(unit_type == UNIT_TYPE_EXTRACTOR,
                                     size=(2, 2))
    candidate_xy = np.transpose(np.nonzero(extractor)).tolist()
    if len(candidate_xy) == 0: return None
    return random.choice(candidate_xy)

# macro actions

def macro_debug(observation):
    micros = [(micro_debug, True)]
    return micros


def macro_do_nothing(observation):
    micros = [(micro_do_nothing, True)]
    return micros


def macro_build_spawning_pool(observation):
    micros = []
    micros.append([micro_move_camera_to_self_base, True])
    micros.append([micro_select_any_worker, True])
    micros.append([micro_build_spawning_pool, True])
    return micros


def macro_build_roach_warren(observation):
    micros = []
    micros.append([micro_move_camera_to_self_base, True])
    micros.append([micro_select_any_worker, True])
    micros.append([micro_build_roach_warren, True])
    return micros


def macro_build_extractor(observation):
    micros = []
    micros.append([micro_move_camera_to_self_base, True])
    micros.append([micro_select_any_worker, True])
    micros.append([micro_build_extractor, True])
    return micros


def macro_train_overlord(observation):
    micros = []
    micros.append((micro_move_camera_to_self_base, True))
    micros.append((micro_select_hatchery, True))
    micros.append((micro_select_larva, True))
    micros.append((micro_train_overlord, True))
    micros.append((micro_train_overlord, False))
    micros.append((micro_train_overlord, False))
    micros.append((micro_move_away_from_hatchery, True))
    return micros


def macro_train_queen(observation):
    micros = []
    micros.append((micro_move_camera_to_self_base, True))
    micros.append((micro_select_hatchery, True))
    micros.append((micro_train_queen, True))
    return micros


def macro_train_worker(observation):
    micros = []
    micros.append((micro_move_camera_to_self_base, True))
    micros.append((micro_select_hatchery, True))
    micros.append((micro_select_larva, True))
    micros.append((micro_train_worker, True))
    micros.append((micro_train_worker, True))
    micros.append((micro_train_worker, True))
    return micros


def macro_train_zergling(observation):
    micros = []
    micros.append((micro_move_camera_to_self_base, True))
    micros.append((micro_select_hatchery, True))
    micros.append((micro_select_larva, True))
    micros.append((micro_train_zergling, True))
    micros.append((micro_train_zergling, True))
    micros.append((micro_train_zergling, True))
    return micros


def macro_train_roach(observation):
    micros = []
    micros.append((micro_move_camera_to_self_base, True))
    micros.append((micro_select_hatchery, True))
    micros.append((micro_select_larva, True))
    micros.append((micro_train_roach, True))
    micros.append((micro_train_roach, True))
    micros.append((micro_train_roach, True))
    return micros


def macro_queen_inject_larva(observation):
    micros = []
    micros.append((micro_move_camera_to_self_base, True))
    micros.append((micro_select_queen, True))
    micros.append((micro_inject_larva, True))
    return micros


def macro_put_aside_overlord(observation):
    micros = [(micro_do_nothing, True)]
    return micros


def macro_move_camera_to_self_base(observation):
    micros = [(micro_move_camera_to_self_base, True)]
    return micros


def macro_move_camera_to_enemy_base(observation):
    micros = [(micro_move_camera_to_enemy_base, True)]
    return micros


def macro_all_idle_workers_collect_minerals(observation):
    micros = []
    micros.append([micro_move_camera_to_self_base, True])
    micros.append([micro_select_all_idle_workers, True])
    micros.append([micro_go_to_minerals, True])
    return micros


def macro_any_worker_collect_vespene(observation):
    micros = []
    micros.append([micro_move_camera_to_self_base, True])
    micros.append([micro_select_any_worker, True])
    micros.append([micro_go_to_extractor, True])
    return micros


def macro_all_defence(observation):
    micros = []
    micros.append([micro_move_camera_to_self_base, True])
    micros.append([micro_select_all_armies, True])
    micros.append([micro_attack_any_screen, True])
    return micros


def macro_all_attack_enemy_base(observation):
    micros = []
    if not has_enemy_screen(observation):
        micros.append([micro_move_camera_to_enemy_base, True])
        micros.append([micro_select_all_armies, True])
        micros.append([micro_attack_center_screen, True])
    else:
        micros.append([micro_select_all_armies, True])
        micros.append([micro_attack_any_screen, True])
    return micros


def macro_all_attack_any_enemy(observation):
    micros = []
    micros.append([micro_move_camera_to_any_enemy, True])
    micros.append([micro_select_all_armies, True])
    micros.append([micro_attack_any_screen, True])
    return micros

# micro actions

def micro_do_nothing(observation):
    function_id = actions.FUNCTIONS.no_op.id
    function_args = []
    return function_id, function_args


def micro_move_camera_to_self_base(observation):
    function_id = actions.FUNCTIONS.move_camera.id
    function_args = [observation["base_xy"][::-1] + [1, 1]]
    return function_id, function_args


def micro_move_camera_to_enemy_base(observation):
    function_id = actions.FUNCTIONS.move_camera.id
    resolution = observation["screen"].shape[-1]
    function_args = [resolution - observation["base_xy"][::-1]]
    return function_id, function_args


def micro_move_camera_to_any_enemy(observation):
    xy = find_enemy_minimap(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.move_camera.id
    function_args = [xy[::-1]]
    return actions.FunctionCall(function_id, function_args)


def micro_select_larva(observation):
    function_id = actions.FUNCTIONS.select_larva.id
    function_args = []
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


def micro_build_spawning_pool(observation):
    if has_spawning_pool_screen(observation):
        return None
    xy = find_vacant_creep_location(observation, 4)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Build_SpawningPool_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_build_roach_warren(observation):
    if has_roach_warren_screen(observation):
        return None
    xy = find_vacant_creep_location(observation, 4)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Build_RoachWarren_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_build_extractor(observation):
    xy = find_vespene_screen(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Build_Extractor_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_train_overlord(observation):
    function_id = actions.FUNCTIONS.Train_Overlord_quick.id
    function_args = [[0]]
    return function_id, function_args


def micro_train_queen(observation):
    function_id = actions.FUNCTIONS.Train_Queen_quick.id
    function_args = [[0]]
    return function_id, function_args


def micro_train_worker(observation):
    function_id = actions.FUNCTIONS.Train_Drone_quick.id
    function_args = [[0]]
    return function_id, function_args


def micro_train_zergling(observation):
    function_id = actions.FUNCTIONS.Train_Zergling_quick.id
    function_args = [[0]]
    return function_id, function_args


def micro_train_roach(observation):
    function_id = actions.FUNCTIONS.Train_Roach_quick.id
    function_args = [[0]]
    return function_id, function_args


def micro_select_hatchery(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    candidate_xy = np.transpose(
        np.nonzero(unit_type == UNIT_TYPE_HATCHERY)).tolist()
    if len(candidate_xy) == 0: return None
    xy = random.choice(candidate_xy)
    function_id = actions.FUNCTIONS.select_point.id
    function_args = [[SELECT_POINT_SELECT], xy[::-1]]
    return function_id, function_args


def micro_select_queen(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    candidate_xy = np.transpose(
        np.nonzero(unit_type == UNIT_TYPE_QUEEN)).tolist()
    if len(candidate_xy) == 0: return None
    xy = random.choice(candidate_xy)
    function_id = actions.FUNCTIONS.select_point.id
    function_args = [[SELECT_POINT_SELECT], xy[::-1]]
    return function_id, function_args


def micro_move_away_from_hatchery(observation):
    xy = find_location_far_from_hatchery(observation, radius=15)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Smart_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_go_to_minerals(observation):
    xy = find_minerals_screen(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Smart_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_go_to_extractor(observation):
    xy = find_extractor_screen(observation)
    if xy is None: return None
    function_id = actions.FUNCTIONS.Smart_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_inject_larva(observation):
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    candidate_xy = np.transpose(
        np.nonzero(unit_type == UNIT_TYPE_HATCHERY)).tolist()
    if len(candidate_xy) == 0: return None
    xy = random.choice(candidate_xy)
    function_id = actions.FUNCTIONS.Effect_InjectLarva_screen.id
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
    xy = [resolution / 2 - 1, resolution / 2 - 1]
    function_id = actions.FUNCTIONS.Attack_screen.id
    function_args = [[NOT_QUEUED], xy[::-1]]
    return function_id, function_args


def micro_debug(observation):
    camera = observation["minimap"][MINIMAP_FEATURES.camera.index]
    unit_type = observation["screen"][SCREEN_FEATURES.unit_type.index]
    np.set_printoptions(threshold=np.nan, linewidth=300)
    print(unit_type)
    function_id = actions.FUNCTIONS.no_op.id
    function_args = []
    return function_id, function_args


def locate_camera_minimap(observation):
    camera = observation["minimap"][MINIMAP_FEATURES.camera.index]
    return np.mean(np.transpose(np.nonzero(camera == 1)), 0).astype(int)


class ZergActionWrapperV0(gym.Wrapper):

    def __init__(self, env):
        super(ZergActionWrapperV0, self).__init__(env)
        assert isinstance(env.action_space, PySC2ActionSpace)
        assert isinstance(env.observation_space, PySC2ObservationSpace)
        shape_screen = self.env.observation_space.space_attr["screen"][1:]
        shape_minimap = self.env.observation_space.space_attr["minimap"][1:]
        assert shape_screen == (32, 32) and shape_minimap == (32, 32)
        self._macro_actions = [macro_do_nothing,
                               macro_train_overlord,
                               macro_train_worker,
                               macro_train_zergling,
                               macro_train_roach,
                               macro_build_spawning_pool,
                               macro_build_roach_warren,
                               macro_build_extractor,
                               macro_any_worker_collect_vespene,
                               macro_all_idle_workers_collect_minerals,
                               macro_all_defence,
                               macro_all_attack_enemy_base,
                               macro_all_attack_any_enemy,
                               macro_train_queen,
                               macro_queen_inject_larva]
        self.action_space = Discrete(len(self._macro_actions))

    def step(self, action):
        macro_action = self._macro_actions[action]
        observation = self._last_observation
        observation["base_xy"] = self._base_xy
        reward_cum = 0
        acted = False
        for micro_action, block in macro_action(observation):
            action = micro_action(observation)
            if (action is None or
                self.env.action_space.contains(
                    action, observation["available_actions"]) == False):
                if block:
                    break
                else:
                    continue
            self._record_action(action)
            observation, reward, done, info = self.env.step(action)
            observation["base_xy"] = self._base_xy
            acted = True
            reward_cum += reward
            if done:
                return observation, reward_cum, done, info
        if not acted:
            action = micro_do_nothing(observation)
            self._record_action(action)
            observation, reward, done, info = self.env.step(action)
            observation["base_xy"] = self._base_xy
            reward_cum += reward
        self._last_observation = observation 
        return observation, reward_cum, done, info
            
    def reset(self):
        observation = self.env.reset()
        self._base_xy = locate_camera_minimap(observation)
        self._num_spawning_pools = 0
        self._num_extractors = 0
        self._num_roach_warrens = 0
        self._num_queens = 0
        self._last_observation = observation
        return observation

    def _record_action(self, action):
        function_id = action[0]
        if function_id == actions.FUNCTIONS.Build_SpawningPool_screen.id:
            self._num_spawning_pools = 1
        if function_id == actions.FUNCTIONS.Build_Extractor_screen.id:
            self._num_extractors = 1
        if function_id == actions.FUNCTIONS.Build_RoachWarren_screen.id:
            self._num_roach_warrens = 1
        if function_id == actions.FUNCTIONS.Train_Queen_quick.id:
            self._num_queens = 1

    @property
    def num_spawning_pools(self):
        return self._num_spawning_pools

    @property
    def num_extractors(self):
        return self._num_extractors

    @property
    def num_roach_warrens(self):
        return self._num_roach_warrens

    @property
    def num_queens(self):
        return self._num_queens
