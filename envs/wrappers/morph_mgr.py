import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.tech_tree import TechTree

from envs.wrappers.utils import Function
from envs.wrappers.const import MAXIMUM_NUM


class MorphManager(object):
    def __init__(self):
        self._tech_tree = TechTree()

    def action(self, func_name, type_id):
        return Function(name=func_name,
                        function=self._morph_unit(type_id),
                        is_valid=self._is_valid_morph_unit(type_id))

    def _morph_unit(self, type_id):

        def act(dc):
            tech = self._tech_tree.getUnitData(type_id)
            morpher = random.choice(dc.idle_units_of_types(tech.whatBuilds))
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(morpher.tag)
            action.action_raw.unit_command.ability_id = tech.buildAbility
            return [action]

        return act

    def _is_valid_morph_unit(self, type_id):

        def is_valid(dc):
            tech = self._tech_tree.getUnitData(type_id)
            # TODO(@xinghai): check requiredUnits and requiredUpgrads
            has_required_units = any([len(dc.mature_units_of_type(u)) > 0
                                      for u in tech.requiredUnits]) \
                                 if len(tech.requiredUnits) > 0 else True
            has_required_upgrades = any([t in dc.upgraded_techs
                                         for t in tech.requiredUpgrades]) \
                                    if len(tech.requiredUpgrades) > 0 else True
            # TODO(@xinghai): add units_with_task here
            overquota = len(dc.units_of_type(type_id)) >= MAXIMUM_NUM[type_id] \
                if type_id in MAXIMUM_NUM else False
            if (dc.mineral_count >= tech.mineralCost and
                dc.gas_count >= tech.gasCost and
                dc.supply_count >= tech.supplyCost and
                has_required_units and
                has_required_upgrades and
                not overquota and
                len(dc.idle_units_of_types(tech.whatBuilds)) > 0):
                return True
            else:
                return False

        return is_valid
