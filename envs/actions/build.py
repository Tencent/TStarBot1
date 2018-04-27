import random
import numpy as np
import math

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.tech_tree import TechTree
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.unit_controls import Unit

from envs.actions.function import Function
import envs.common.utils as utils
from envs.common.const import AREA_COLLISION_BUILDINGS
from envs.common.const import MAXIMUM_NUM


class BuildActions(object):
    def __init__(self):
        self._spatial_planner = SpatialPlanner()
        self._tech_tree = TechTree()

    def action(self, func_name, type_id):
        return Function(name=func_name,
                        function=self._build_unit(type_id),
                        is_valid=self._is_valid_build_unit(type_id))

    def _build_unit(self, type_id):

        def act(dc):
            tech = self._tech_tree.getUnitData(type_id)
            pos = self._spatial_planner.get_building_position(type_id, dc)
            builder = utils.closest_unit(pos, dc.units_of_types(tech.whatBuilds))
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(builder.tag)
            action.action_raw.unit_command.ability_id = tech.buildAbility
            if isinstance(pos, Unit):
                action.action_raw.unit_command.target_unit_tag = pos.tag
            else:
                action.action_raw.unit_command.target_world_space_pos.x = pos[0]
                action.action_raw.unit_command.target_world_space_pos.y = pos[1]
            return [action]

        return act

    def _is_valid_build_unit(self, type_id):

        def is_valid(dc):
            tech = self._tech_tree.getUnitData(type_id)
            # TODO(@xinghai): check requiredUnits and requiredUpgrads
            has_required_units = any([len(dc.mature_units_of_type(u)) > 0
                                      for u in tech.requiredUnits]) \
                                 if len(tech.requiredUnits) > 0 else True
            has_required_upgrades = any([t in dc.upgraded_techs
                                         for t in tech.requiredUpgrades]) \
                                    if len(tech.requiredUpgrades) > 0 else True
            current_num = len(dc.units_of_type(type_id)) + \
                len(dc.units_with_task(tech.buildAbility))
            overquota = current_num >= MAXIMUM_NUM[type_id] \
                if type_id in MAXIMUM_NUM else False
            if (dc.mineral_count >= tech.mineralCost and
                dc.gas_count >= tech.gasCost and
                dc.supply_count >= tech.supplyCost and
                has_required_units and
                has_required_upgrades and
                not overquota and
                len(dc.units_of_types(tech.whatBuilds)) > 0 and
                len(dc.units_with_task(tech.buildAbility)) == 0 and
                self._spatial_planner.can_build(type_id, dc)):
                return True
            else:
                return False

        return is_valid


class SpatialPlanner(object):

    def get_building_position(self, type_id, dc):
        if type_id == UNIT_TYPE.ZERG_HATCHERY.value:
            return self._next_base_area(dc)
        elif type_id == UNIT_TYPE.ZERG_EXTRACTOR.value:
            gas = dc.exploitable_gas
            return random.choice(gas) if len(gas) > 0 else None
        else:
            areas = self._constructable_areas(2, dc)
            return random.choice(areas) if len(areas) > 0 else None

    def can_build(self, type_id, dc):
        if type_id == UNIT_TYPE.ZERG_HATCHERY.value:
            return self._next_base_area(dc) is not None
        elif type_id == UNIT_TYPE.ZERG_EXTRACTOR.value:
            return len(dc.exploitable_gas) > 0
        else:
            areas = self._constructable_areas(2, dc)
            # TODO(@xinghai): to be replaced
            if len(areas) > 0: random.choice(areas)
            return len(self._constructable_areas(2, dc)) > 0

    def _constructable_areas(self, margin, dc):
        areas = []
        bases = dc.mature_units_of_types([UNIT_TYPE.ZERG_HATCHERY.value,
                                          UNIT_TYPE.ZERG_LAIR.value,
                                          UNIT_TYPE.ZERG_HIVE.value])
        for base in bases:
            search_region = (base.float_attr.pos_x - 10,
                             base.float_attr.pos_y - 10,
                             10 * 2,
                             10 * 2)
            areas.extend(self._search_areas(
                search_region, dc, margin=margin, remove_corner=True))
        return areas

    def _next_base_area(self, dc):
        unexploited_minerals = dc.unexploited_minerals
        if len(unexploited_minerals) == 0:
            return None
        mineral_to_exploit = utils.closest_unit(dc.init_base_pos,
                                                unexploited_minerals)
        resources_nearby = utils.units_nearby(mineral_to_exploit,
                                              dc.minerals + dc.gas,
                                              max_distance=15)
        x_list = [u.float_attr.pos_x for u in resources_nearby]
        y_list = [u.float_attr.pos_y for u in resources_nearby]
        x_mean = sum(x_list) / len(x_list)
        y_mean = sum(y_list) / len(y_list)
        x_center = (max(x_list) + min(x_list)) / 2
        y_center = (max(y_list) + min(y_list)) / 2
        x_offset = -2 if x_mean < x_center else -6
        y_offset = -2 if y_mean < y_center else -6
        topdown = [min(x_list) + x_offset, min(y_list) + y_offset]
        size = [max(x_list) - min(x_list) + 8, max(y_list) - min(y_list) + 8]
        region = topdown + size
        areas = self._search_areas(region, dc, margin=5)
        return utils.closest_unit((x_center, y_center), areas) \
               if len(areas) > 0 else None

    def _search_areas(self, search_region, dc, margin=0, remove_corner=False):
        bottomleft = tuple(map(int, search_region[:2]))
        size = tuple(map(int, search_region[2:]))
        grids = np.zeros(size).astype(np.int)
        if remove_corner:
            cx, cy = size[0] / 2.0, size[1] / 2.0
            r = max(cx, cy)
            for x in range(size[0]):
                for y in range(size[1]):
                    if (x - cx) ** 2 + (y - cy) ** 2 > r ** 2:
                        grids[x, y] = 1
        for u in dc.units:
            if u.unit_type in AREA_COLLISION_BUILDINGS:
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
