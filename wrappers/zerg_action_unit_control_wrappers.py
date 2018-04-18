import gym
import platform
import numpy as np
import random
import math
from enum import Enum, unique
from collections import namedtuple
from collections import defaultdict

from pysc2.lib import actions as pysc2_actions
from pysc2.lib.unit_controls import Unit
from pysc2.lib.typeenums import UNIT_TYPEID, ABILITY_ID
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb

from envs.space import PySC2RawObservation
from envs.space import MaskableDiscrete


PLAYERINFO_MINERAL_COUNT = 1
PLAYERINFO_VESPENE_COUNT = 2
PLAYERINFO_FOOD_USED = 3
PLAYERINFO_FOOD_CAP = 4
PLAYERINFO_FOOD_ARMY = 5
PLAYERINFO_FOOD_WORKER = 6
PLAYERINFO_IDLE_WORKER_COUNT = 7
PLAYERINFO_LARVA_COUNT = 10

PLACE_COLLISION_UNIT_SET = {UNIT_TYPEID.ZERG_HATCHERY.value,
                            UNIT_TYPEID.ZERG_LAIR.value,
                            UNIT_TYPEID.ZERG_SPAWNINGPOOL.value,
                            UNIT_TYPEID.ZERG_ROACHWARREN.value,
                            UNIT_TYPEID.ZERG_HYDRALISKDEN.value,
                            UNIT_TYPEID.ZERG_EXTRACTOR.value,
                            UNIT_TYPEID.NEUTRAL_MINERALFIELD.value,
                            UNIT_TYPEID.NEUTRAL_MINERALFIELD750.value,
                            UNIT_TYPEID.NEUTRAL_VESPENEGEYSER.value}

AIR_COMBAT_UNIT_SET = {}
LAND_COMBAT_UNIT_SET = {UNIT_TYPEID.ZERG_ROACH.value,
                        UNIT_TYPEID.ZERG_ZERGLING.value}
AIR_LAND_COMBAT_UNIT_SET = {UNIT_TYPEID.ZERG_HYDRALISK.value}


@unique
class AllianceType(Enum):
    SELF = 1
    ALLY = 2
    NEUTRAL = 3
    ENEMY = 4


class ZergData(object):
    def __init__(self):
        self._units = None
        self._player_info = None
        self._attacking_tags = set()

    def update(self, observation):
        self._units = observation['units']
        self._player_info = observation['player']

    def reset(self, observation):
        self._attacking_tags.clear()
        self.update(observation)
        self._init_base_pos = (self.bases[0].float_attr.pos_x,
                               self.bases[0].float_attr.pos_y)

    def label_attack_status(self, units):
        for u in units:
            self._attacking_tags.add(u.tag)

    def get_pos_to_build(self, margin=3):
        cand_pos = []
        for h in self.mature_bases:
            region = (h.float_attr.pos_x - 9, h.float_attr.pos_y - 9,
                      9 * 2, 9* 2)
            cand_pos.extend(self._find_empty_place(region, margin=margin))
        return random.choice(cand_pos) if len(cand_pos) > 0 else None

    def get_pos_to_base(self):
        unexploited_minerals = [
            u for u in self.minerals
            if self._closest_distance(u, self.bases) > 20]
        if len(unexploited_minerals) == 0:
            return None
        mineral_seed = self._closest_units(
            self._init_base_pos, unexploited_minerals, num=1)[0]
        resources_in_range = self._units_in_range(
            mineral_seed, self.minerals + self.vespenes, max_distance=15)
        x_list = [u.float_attr.pos_x for u in resources_in_range]
        y_list = [u.float_attr.pos_y for u in resources_in_range]
        x_mean = sum(x_list) / len(x_list)
        y_mean = sum(y_list) / len(y_list)
        x_center = (max(x_list) + min(x_list)) / 2
        y_center = (max(y_list) + min(y_list)) / 2
        x_offset = -2 if x_mean < x_center else -6
        y_offset = -2 if y_mean < y_center else -6
        topdown = [min(x_list) + x_offset, min(y_list) + y_offset]
        size = [max(x_list) - min(x_list) + 8, max(y_list) - min(y_list) + 8]
        region = topdown + size
        cand_pos = self._find_empty_place(region, margin=5)
        return self._closest_units((x_center, y_center), cand_pos, num=1)[0] \
               if len(cand_pos) > 0 else None

    def closest_drones(self, unit, num=1):
        return self._closest_units(unit, self.drones, num=num)

    def closest_hatcheries(self, unit, num=1):
        return self._closest_units(unit, self.hatcheries, num=num)

    def closest_mature_hatcheries(self, unit, num=1):
        return self._closest_units(unit, self.mature_hatcheries, num=num)

    def closest_bases(self, unit, num=1):
        return self._closest_units(unit, self.bases, num=num)

    def closest_mature_bases(self, unit, num=1):
        return self._closest_units(unit, self.mature_bases, num=num)

    def closest_minerals(self, unit, num=1):
        return self._closest_units(unit, self.minerals, num=num)

    def closest_enemy_units(self, unit, num=1):
        return self._closest_units(unit, self.enemy_units, num=num)

    def closest_enemy_groups(self, unit, num=1):
        assert num >= 1
        return sorted(
            self.enemy_groups,
            key=lambda g: self._distance(unit, self._centroid(g)))[:num]

    @property
    def init_base_pos(self):
        return self._init_base_pos

    @property
    def mineral_count(self):
        return self._player_info[PLAYERINFO_MINERAL_COUNT]

    @property
    def vespene_count(self):
        return self._player_info[PLAYERINFO_VESPENE_COUNT]

    @property
    def resources(self):
        return minerals + vespenes

    @property
    def minerals(self):
        return [u for u in self._units
                if (u.unit_type == UNIT_TYPEID.NEUTRAL_MINERALFIELD.value or
                    u.unit_type == UNIT_TYPEID.NEUTRAL_MINERALFIELD750.value)]

    @property
    def vespenes(self):
        return [u for u in self._units
                if u.unit_type == UNIT_TYPEID.NEUTRAL_VESPENEGEYSER.value]

    @property
    def exploitable_vespenes(self):
        extractors = self.extractors + self.enemy_extractors
        bases = self.mature_bases
        return [u for u in self.vespenes
                if (self._closest_distance(u, bases) < 10 and
                    self._closest_distance(u, extractors) > 3)]

    @property
    def larvas(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_LARVA.value)]

    @property
    def drones(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_DRONE.value)]

    @property
    def idle_drones(self):
        return [u for u in self.drones if len(u.orders) == 0]

    @property
    def hatcheries(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_HATCHERY.value)]

    @property
    def mature_hatcheries(self):
        return [u for u in self.hatcheries
                if u.float_attr.build_progress == 1.0]

    @property
    def idle_hatcheries(self):
        return [u for u in self.mature_hatcheries if len(u.orders) == 0]

    @property
    def lairs(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_LAIR.value)]
    @property
    def idle_lairs(self):
        return [u for u in self.lairs if len(u.orders) == 0]

    @property
    def bases(self):
        return self.hatcheries + self.lairs

    @property
    def mature_bases(self):
        return self.mature_hatcheries + self.lairs

    @property
    def idle_bases(self):
        return self.idle_hatcheries + self.idle_lairs

    @property
    def overlords(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value)]

    @property
    def zerglings(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_ZERGLING.value)]

    @property
    def roaches(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_ROACH.value)]

    @property
    def hydralisks(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_HYDRALISK.value)]

    @property
    def queens(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_QUEEN.value)]

    @property
    def larva_injectable_queens(self):
        return [u for u in self.queens if u.float_attr.energy >= 25]

    @property
    def combat_units(self):
        return self.zerglings + self.roaches + self.hydralisks

    @property
    def idle_combat_units(self):
        return [u for u in self.combat_units if len(u.orders) == 0]

    @property
    def air_combat_units(self):
        return []

    @property
    def land_combat_units(self):
        return self.zerglings + self.roaches

    @property
    def air_land_combat_units(self):
        return self.hydralisks

    @property
    def attacking_combat_units(self):
        return [u for u in self.combat_units if u.tag in self._attacking_tags]

    @property
    def food(self):
        return self._player_info[PLAYERINFO_FOOD_CAP] - \
            self._player_info[PLAYERINFO_FOOD_USED]

    @property
    def has_drone_move_to_build_hatchery(self):
        for u in self.drones:
            if (len(u.orders) > 0 and
                u.orders[0].ability_id == ABILITY_ID.BUILD_HATCHERY.value):
                return True
        return False

    @property
    def has_drone_move_to_build_spawning_pool(self):
        for u in self.drones:
            if (len(u.orders) > 0 and
                u.orders[0].ability_id == ABILITY_ID.BUILD_SPAWNINGPOOL.value):
                return True
        return False

    @property
    def has_drone_move_to_build_extractor(self):
        for u in self.drones:
            if (len(u.orders) > 0 and
                u.orders[0].ability_id == ABILITY_ID.BUILD_EXTRACTOR.value):
                return True
        return False

    @property
    def has_drone_move_to_build_roach_warren(self):
        for u in self.drones:
            if (len(u.orders) > 0 and
                u.orders[0].ability_id == ABILITY_ID.BUILD_ROACHWARREN.value):
                return True
        return False

    @property
    def has_drone_move_to_build_hydraliskden(self):
        for u in self.drones:
            if (len(u.orders) > 0 and
                u.orders[0].ability_id == ABILITY_ID.BUILD_HYDRALISKDEN.value):
                return True
        return False

    @property
    def extractors(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_EXTRACTOR.value)]

    @property
    def mature_extractors(self):
        return [u for u in self.extractors
                if u.float_attr.build_progress == 1.0]

    @property
    def notbusy_extractors(self):
        return [u for u in self.mature_extractors
                if u.int_attr.ideal_harvesters - u.int_attr.assigned_harvesters]

    @property
    def spawning_pools(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_SPAWNINGPOOL.value)]

    @property
    def mature_spawning_pools(self):
        return [u for u in self.spawning_pools
                if u.float_attr.build_progress == 1.0]

    @property
    def roach_warrens(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_ROACHWARREN.value)]

    @property
    def mature_roach_warrens(self):
        return [u for u in self.roach_warrens
                if u.float_attr.build_progress == 1.0]

    @property
    def hydraliskdens(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.SELF.value and
                    u.unit_type == UNIT_TYPEID.ZERG_HYDRALISKDEN.value)]

    @property
    def mature_hydraliskdens(self):
        return [u for u in self.hydraliskdens
                if u.float_attr.build_progress == 1.0]

    @property
    def is_any_unit_selected(self):
        for u in self._units:
            if u.bool_attr.is_selected:
                return True
        return False

    @property
    def enemy_extractors(self):
        return [u for u in self._units
                if (u.int_attr.alliance == AllianceType.ENEMY.value and
                    u.unit_type == UNIT_TYPEID.ZERG_EXTRACTOR.value)]

    @property
    def enemy_units(self):
        return [u for u in self._units
                if u.int_attr.alliance == AllianceType.ENEMY.value]

    @property
    def enemy_groups(self):
        groups = defaultdict(list)
        for u in self.enemy_units:
            grid_x, grid_y = u.float_attr.pos_x // 30, u.float_attr.pos_y // 30
            groups[(grid_x, grid_y)].append(u)
        return list(groups.values())

    def _closest_units(self, unit, target_units, num=1):
        assert num >= 1
        if num == 1:
            return [min(target_units, key=lambda u: self._distance(unit, u))] \
                   if len(target_units) > 0 else []
        else:
            return sorted(target_units,
                          key=lambda u: self._distance(unit, u))[:num]

    def _closest_distance(self, unit, target_units):
        return min(self._distance(unit, u) for u in target_units) \
               if len(target_units) > 0 else float('inf')

    def _units_in_range(self, unit, target_units, max_distance):
        return [u for u in target_units
                if self._distance(unit, u) <= max_distance]

    def _distance(self, a, b):
        if isinstance(a, Unit) and isinstance(b, Unit):
            return ((a.float_attr.pos_x - b.float_attr.pos_x) ** 2 +
                    (a.float_attr.pos_y - b.float_attr.pos_y) ** 2) ** 0.5
        elif not isinstance(a, Unit) and isinstance(b, Unit):
            return ((a[0] - b.float_attr.pos_x) ** 2 +
                    (a[1] - b.float_attr.pos_y) ** 2) ** 0.5
        elif isinstance(a, Unit) and not isinstance(b, Unit):
            return ((a.float_attr.pos_x - b[0]) ** 2 +
                    (a.float_attr.pos_y - b[1]) ** 2) ** 0.5
        else:
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _centroid(self, units):
        assert len(units) > 0
        return (sum(u.float_attr.pos_x for u in units) / len(units),
                sum(u.float_attr.pos_y for u in units) / len(units))

    def _find_empty_place(self, search_region, margin=0):
        bottomleft = tuple(map(int, search_region[:2]))
        size = tuple(map(int, search_region[2:]))
        grids = np.zeros(size).astype(np.int)
        for u in self._units:
            if u.unit_type in PLACE_COLLISION_UNIT_SET:
                radius = (u.float_attr.radius // 0.5) * 0.5 + margin
                xl = int(math.floor(u.float_attr.pos_x - radius)) - bottomleft[0]
                xr = int(math.ceil(u.float_attr.pos_x + radius)) - bottomleft[0]
                yu = int(math.floor(u.float_attr.pos_y - radius)) - bottomleft[1]
                yd = int(math.ceil(u.float_attr.pos_y + radius)) - bottomleft[1]
                for x in range(max(xl, 0), min(xr, size[0])):
                    for y in range(max(yu, 0), min(yd, size[1])):
                        grids[x, y] = 1
        x, y = np.nonzero(1 - grids)
        #np.set_printoptions(threshold=np.nan, linewidth=300)
        #print(grids)
        return list(zip(x + bottomleft[0] + 0.5, y + bottomleft[1] + 0.5))


Function = namedtuple('Function', ['name', 'function', 'is_valid'])


class ActionCreator(object):

    @staticmethod
    def attack(who, target=None, pos=None):
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.extend([u.tag for u in who])
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.ATTACK.value
        if target is not None:
            action.action_raw.unit_command.target_unit_tag = target.tag
        if pos is not None:
            action.action_raw.unit_command.target_world_space_pos.x = pos[0]
            action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        return action

    @staticmethod
    def harvest_gather(who, target):
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.extend([u.tag for u in who])
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.HARVEST_GATHER_DRONE.value
        action.action_raw.unit_command.target_unit_tag = target.tag
        return action


class ZergActionWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ZergActionWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)
        self._data = ZergData()
        self._actions = [
            Function(
                name='idle', # 0
                function=self._idle,
                is_valid=self._is_valid_idle),
            Function(
                name='build_extractor', # 1
                function=self._build_extractor,
                is_valid=self._is_valid_build_extractor),
            Function(
                name='build_spawning_pool', # 2
                function=self._build_spawning_pool,
                is_valid=self._is_valid_build_spawning_pool),
            Function(
                name='build_roach_warren', # 3
                function=self._build_roach_warren,
                is_valid=self._is_valid_build_roach_warren),
            Function(
                name='build_hydraliskden', # 4
                function=self._build_hydraliskden,
                is_valid=self._is_valid_build_hydraliskden),
            Function(
                name='build_hatchery', # 5
                function=self._build_hatchery,
                is_valid=self._is_valid_build_hatchery),
            Function(
                name='produce_overlord', # 6
                function=self._produce_overlord,
                is_valid=self._is_valid_produce_overlord),
            Function(
                name='produce_drone', # 7
                function=self._produce_drone,
                is_valid=self._is_valid_produce_drone),
            Function(
                name='produce_zergling', # 8
                function=self._produce_zergling,
                is_valid=self._is_valid_produce_zergling),
            Function(
                name='produce_roach', # 9
                function=self._produce_roach,
                is_valid=self._is_valid_produce_roach),
            Function(
                name='produce_queen', # 10
                function=self._produce_queen,
                is_valid=self._is_valid_produce_queen),
            Function(
                name='produce_hydralisk', # 11
                function=self._produce_hydralisk,
                is_valid=self._is_valid_produce_hydralisk),
            Function(
                name='inject_larva', # 12
                function=self._inject_larva,
                is_valid=self._is_valid_inject_larva),
            Function(
                name='assign_drones_to_extractor', # 13
                function=self._assign_drones_to_extractor,
                is_valid=self._is_valid_assign_drones_to_extractor),
            Function(
                name='morph_lair', # 14
                function=self._morph_lair,
                is_valid=self._is_valid_morph_lair),
            Function(
                name='rally_idle_combat_units', # 15
                function=self._rally_idle_combat_units,
                is_valid=self._is_valid_rally_idle_combat_units),
            Function(
                name='attack_closest_unit_30', # 16
                function=self._attack_closest_unit,
                is_valid=self._is_valid_attack_30),
            Function(
                name='attack_closest_unit_20', # 17
                function=self._attack_closest_unit,
                is_valid=self._is_valid_attack_20),
        ]
        self.action_space = MaskableDiscrete(len(self._actions))

    def step(self, action):
        '''
        if self._action_mask[action] == 0:
            print("%s Not Available" % self._actions[action].name)
            print("Availables:")
            for i, act in enumerate(self._actions):
                if self._action_mask[i] == 1:
                    print("%d %s", (i, act.name))
            action = 0
        '''
        assert self._action_mask[action] == 1
        actions = self._actions[action].function()
        if self._is_valid_assign_idle_drones_to_minerals():
            actions = self._assign_idle_drones_to_minerals() + actions
        if platform.system() != 'Linux' and not self._data.is_any_unit_selected:
            actions = self._select_all() + actions
        observation, reward, done, info = self.env.step(actions)
        self._data.update(observation)
        self._action_mask = self._get_valid_action_mask()
        return (observation, self._action_mask), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._data.reset(observation)
        self._action_mask = self._get_valid_action_mask()
        return (observation, self._action_mask)

    @property
    def player_position(self):
        if self._data.init_base_pos[0] < 100:
            return 0
        else:
            return 1

    def _get_valid_action_mask(self):
        ids = [i for i, action in enumerate(self._actions) if action.is_valid()]
        mask = np.zeros(self.action_space.n)
        mask[ids] = 1
        return mask

    def _idle(self):
        return []

    def _is_valid_idle(self):
        return True

    def _select_all(self):
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = 0
        select = action.action_feature_layer.unit_selection_rect
        out_rect = select.selection_screen_coord.add()
        point.Point(0, 0).assign_to(out_rect.p0)
        point.Point(32, 32).assign_to(out_rect.p1)
        select.selection_add = True
        actions.append(action)
        return actions

    def _build_extractor(self):
        vespene = random.choice(self._data.exploitable_vespenes)
        drone = self._data.closest_drones(vespene, num=1)[0]
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.BUILD_EXTRACTOR.value
        action.action_raw.unit_command.unit_tags.append(drone.tag)
        action.action_raw.unit_command.target_unit_tag = vespene.tag
        actions.append(action)
        return actions

    def _is_valid_build_extractor(self):
        if (len(self._data.exploitable_vespenes) > 0 and
            not self._data.has_drone_move_to_build_extractor and
            self._data.mineral_count >= 25 and
            len(self._data.drones) >= 1):
            return True
        else:
            return False

    def _build_spawning_pool(self):
        pos = self._data.get_pos_to_build(margin=3)
        drone = self._data.closest_drones(pos, num=1)[0]
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.BUILD_SPAWNINGPOOL.value
        action.action_raw.unit_command.unit_tags.append(drone.tag)
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        actions.append(action)
        return actions

    def _is_valid_build_spawning_pool(self):
        if (len(self._data.spawning_pools) == 0 and
            not self._data.has_drone_move_to_build_spawning_pool and
            self._data.mineral_count >= 200 and
            self._data.get_pos_to_build(margin=3) is not None and
            len(self._data.drones) > 0):
            return True
        else:
            return False

    def _build_hatchery(self):
        pos = self._data.get_pos_to_base()
        drone = self._data.closest_drones(pos, num=1)[0]
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.BUILD_HATCHERY.value
        action.action_raw.unit_command.unit_tags.append(drone.tag)
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        actions.append(action)
        return actions

    def _is_valid_build_hatchery(self):
        if (self._data.mineral_count >= 300 and
            not self._data.has_drone_move_to_build_hatchery and
            self._data.get_pos_to_base() is not None and
            len(self._data.drones) > 0):
            return True
        else:
            return False

    def _build_roach_warren(self):
        pos = self._data.get_pos_to_build(margin=3)
        drone = self._data.closest_drones(pos, num=1)[0]
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.BUILD_ROACHWARREN.value
        action.action_raw.unit_command.unit_tags.append(drone.tag)
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        actions.append(action)
        return actions

    def _is_valid_build_roach_warren(self):
        if (len(self._data.mature_spawning_pools) > 0 and
            len(self._data.roach_warrens) == 0 and
            not self._data.has_drone_move_to_build_roach_warren and
            self._data.mineral_count >= 150 and
            self._data.get_pos_to_build(margin=3) is not None and
            len(self._data.drones) > 0):
            return True
        else:
            return False

    def _build_hydraliskden(self):
        pos = self._data.get_pos_to_build(margin=3)
        drone = self._data.closest_drones(pos, num=1)[0]
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.BUILD_HYDRALISKDEN.value
        action.action_raw.unit_command.unit_tags.append(drone.tag)
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        actions.append(action)
        return actions

    def _is_valid_build_hydraliskden(self):
        if (len(self._data.lairs) > 0 and
            len(self._data.hydraliskdens) == 0 and
            not self._data.has_drone_move_to_build_hydraliskden and
            self._data.mineral_count >= 100 and
            self._data.vespene_count >= 100 and
            self._data.get_pos_to_build(margin=3) is not None and
            len(self._data.drones) > 0):
            return True
        else:
            return False

    def _produce_drone(self):
        actions = []
        larva = random.choice(self._data.larvas)
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(larva.tag)
        action.action_raw.unit_command.ability_id = ABILITY_ID.TRAIN_DRONE.value
        actions.append(action)
        return actions

    def _is_valid_produce_drone(self):
        if (self._data.mineral_count >= 50 and
            self._data.food >= 1 and
            len(self._data.larvas) > 0):
            return True
        else:
            return False

    def _produce_overlord(self):
        larva = random.choice(self._data.larvas)
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(larva.tag)
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.TRAIN_OVERLORD.value
        actions.append(action)
        return actions

    def _is_valid_produce_overlord(self):
        if (self._data.mineral_count >= 100 and
            len(self._data.larvas) > 0):
            return True
        else:
            return False

    def _produce_zergling(self):
        larva = random.choice(self._data.larvas)
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(larva.tag)
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.TRAIN_ZERGLING.value
        actions.append(action)
        return actions

    def _is_valid_produce_zergling(self):
        if (self._data.mineral_count >= 50 and
            len(self._data.larvas) > 0 and
            self._data.food >= 1 and
            len(self._data.mature_spawning_pools) > 0):
            return True
        else:
            return False

    def _produce_roach(self):
        larva = random.choice(self._data.larvas)
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(larva.tag)
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.TRAIN_ROACH.value
        actions.append(action)
        return actions

    def _is_valid_produce_roach(self):
        if (self._data.mineral_count >= 75 and
            self._data.vespene_count >= 25 and
            len(self._data.larvas) > 0 and
            self._data.food >= 2 and
            len(self._data.mature_roach_warrens) > 0):
            return True
        else:
            return False

    def _produce_hydralisk(self):
        larva = random.choice(self._data.larvas)
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(larva.tag)
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.TRAIN_HYDRALISK.value
        actions.append(action)
        return actions

    def _is_valid_produce_hydralisk(self):
        if (self._data.mineral_count >= 100 and
            self._data.vespene_count >= 50 and
            len(self._data.larvas) > 0 and
            self._data.food >= 2 and
            len(self._data.mature_hydraliskdens) > 0):
            return True
        else:
            return False

    def _assign_drones_to_extractor(self):
        actions = []
        extractor = random.choice(self._data.notbusy_extractors)
        num_need = extractor.int_attr.ideal_harvesters - \
            extractor.int_attr.assigned_harvesters
        assert num_need > 0
        drones = self._data.closest_drones(extractor, num=num_need)
        assert len(drones) > 0
        actions.append(ActionCreator.harvest_gather(drones, extractor))
        return actions

    def _is_valid_assign_drones_to_extractor(self):
        if (len(self._data.notbusy_extractors) > 0 and
            len(self._data.drones) > 0):
            return True
        else:
            return False

    def _assign_idle_drones_to_minerals(self):
        actions = []
        for drone in self._data.idle_drones:
            mineral = self._data.closest_minerals(drone, num=1)[0]
            actions.append(ActionCreator.harvest_gather([drone], mineral))
        return actions

    def _is_valid_assign_idle_drones_to_minerals(self):
        if (len(self._data.idle_drones) > 0 and
            len(self._data.minerals) > 0):
            return True
        else:
            return False

    def _produce_queen(self):
        base = random.choice(self._data.mature_bases)
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(base.tag)
        action.action_raw.unit_command.ability_id = \
            ABILITY_ID.TRAIN_QUEEN.value
        actions.append(action)
        return actions

    def _is_valid_produce_queen(self):
        if (self._data.mineral_count >= 150 and
            self._data.food >= 2 and
            len(self._data.idle_bases) > 0 and
            len(self._data.mature_spawning_pools) > 0):
            return True
        else:
            return False

    def _inject_larva(self):
        actions = []
        for queen in self._data.larva_injectable_queens:
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(queen.tag)
            action.action_raw.unit_command.ability_id = \
                ABILITY_ID.EFFECT_INJECTLARVA.value
            base = self._data.closest_mature_bases(queen, num=1)[0]
            action.action_raw.unit_command.target_unit_tag = base.tag
            actions.append(action)
        return actions

    def _is_valid_inject_larva(self):
        if (len(self._data.mature_bases) > 0 and
            len(self._data.larva_injectable_queens) > 0):
            return True
        else:
            return False

    def _morph_lair(self):
        hatchery = random.choice(self._data.idle_hatcheries)
        actions = []
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(hatchery.tag)
        action.action_raw.unit_command.ability_id = ABILITY_ID.MORPH_LAIR.value
        actions.append(action)
        return actions

    def _is_valid_morph_lair(self):
        if (self._data.mineral_count >= 150 and
            self._data.vespene_count >= 100 and
            len(self._data.mature_spawning_pools) > 0 and
            len(self._data.idle_hatcheries) > 0):
            return True
        else:
            return False

    def _attack_closest_unit(self):
        return self._micro_attack(self._data.combat_units,
                                  self._data.enemy_units)

    def _is_valid_attack_30(self):
        if (len(self._data.combat_units) > 30 and
            len(self._data.enemy_units) > 0):
            return True
        else:
            return False

    def _is_valid_attack_20(self):
        if (len(self._data.combat_units) > 20 and
            len(self._data.enemy_units) > 0):
            return True
        else:
            return False

    def _attack_closest_group(self):
        enemy_group = self._data.closest_enemy_groups(
            self._data.init_base_pos)[0]
        return self._micro_attack(enemy_group)

    def _rally_idle_combat_units(self):
        if self.player_position == 0:
            rally_pos = (65, 113)
        else:
            rally_pos = (138, 36)
        return [ActionCreator.attack(self._data.idle_combat_units,
                                     pos=rally_pos)]

    def _is_valid_rally_idle_combat_units(self):
        if len(self._data.idle_combat_units) > 0:
            return True
        else:
            return False

    def _micro_attack(self, self_units, enemy_units):

        def select_and_attack(unit, enemy_units):
            assert len(enemy_units) > 0
            close_units = self._closest_units(unit, enemy_units, 5)
            weakest_unit = self._weakest_units(unit, close_units, 1)[0]
            return ActionCreator.attack([unit], target=weakest_unit)

        air_combat_units = [u for u in self_units
                            if u.unit_type in AIR_COMBAT_UNIT_SET]
        land_combat_units = [u for u in self_units
                             if u.unit_type in LAND_COMBAT_UNIT_SET]
        air_land_combat_units = [u for u in self_units
                                 if u.unit_type in AIR_LAND_COMBAT_UNIT_SET]
        air_enemy_units = [u for u in enemy_units if u.bool_attr.is_flying]
        land_enemy_units = [u for u in enemy_units if not u.bool_attr.is_flying]
        actions = []
        for unit in air_combat_units:
            if len(air_enemy_units) > 0:
                actions.append(select_and_attack(unit, air_enemy_units))
        for unit in land_combat_units:
            if len(land_enemy_units) > 0:
                actions.append(select_and_attack(unit, land_enemy_units))
        for unit in air_land_combat_units:
            if len(air_enemy_units) > 0:
                actions.append(select_and_attack(unit, air_enemy_units))
            if len(land_enemy_units) > 0:
                actions.append(select_and_attack(unit, land_enemy_units))
        return actions

    def _closest_units(self, unit, target_units, num=1):
        assert num >= 1
        if num == 1:
            return [min(target_units, key=lambda u: self._distance(unit, u))] \
                   if len(target_units) > 0 else []
        else:
            return sorted(target_units,
                          key=lambda u: self._distance(unit, u))[:num]

    def _weakest_units(self, unit, target_units, num=1):
        assert num >= 1
        if num == 1:
            return [min(target_units, key=lambda u: u.float_attr.health)] \
                   if len(target_units) > 0 else []
        else:
            return sorted(target_units,
                          key=lambda u: u.float_attr.health)[:num]

    def _weakest_unit(self, unit, target_units):
        assert len(target_units) > 0
        return min(target_units, key=lambda u: u.float_attr.health)

    def _distance(self, a, b):
        return ((a.float_attr.pos_x - b.float_attr.pos_x) ** 2 +
                (a.float_attr.pos_y - b.float_attr.pos_y) ** 2) ** 0.5
