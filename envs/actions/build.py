from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.tech_tree import TechTree
from pysc2.lib.unit_controls import Unit

from envs.actions.function import Function
from envs.actions.spatial_planner import SpatialPlanner
import envs.common.utils as utils
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