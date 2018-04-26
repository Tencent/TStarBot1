import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import ABILITY_ID as ABILITY

from envs.actions.function import Function
import envs.common.utils as utils
from envs.common.const import ATTACK_FORCE
from envs.common.const import ALLY_TYPE


class CombatActions(object):

    def __init__(self):
        #TODO: add more combat types
        self._combat_types = [UNIT_TYPE.ZERG_ZERGLING.value,
                              UNIT_TYPE.ZERG_ROACH.value,
                              UNIT_TYPE.ZERG_HYDRALISK.value]

        self._attack_unit_tags = set()

    def reset(self):
        self._attack_unit_tags.clear()

    @property
    def action_rally_new_combat_units(self):
        return Function(
            name="rally_new_combat_units",
            function=self._rally_new_combat_units,
            is_valid=self._is_valid_rally_new_combat_units)

    @property
    def action_rally_idle_combat_units_to_midfield(self):
        return Function(
            name="rally_idle_combat_units_to_midfield",
            function=self._rally_idle_combat_units_to_midfield,
            is_valid=self._is_valid_rally_idle_combat_units_to_midfield)

    @property
    def action_all_attack_30(self):
        return Function(
            name="all_attack_30",
            function=self._all_attack_closest_unit,
            is_valid=self._is_valid_all_attack_closest_unit_30)

    @property
    def action_all_attack_20(self):
        return Function(
            name="all_attack_20",
            function=self._all_attack_closest_unit,
            is_valid=self._is_valid_all_attack_closest_unit_20)

    @property
    def action_universal_micro_attack(self):
        return Function(
            name="all_attack_20",
            function=self._all_units_of_attack_status_do_micro_attack,
            is_valid=self._is_valid_all_units_of_attack_do_micro_attack)

    def _rally_new_combat_units(self, dc):
        if dc.init_base_pos[0] < 100:
            rally_pos = (68, 108)
        else:
            rally_pos = (133, 36)
        new_combat_units = [u for u in dc.units_of_types(self._combat_types)
                            if dc.is_new_unit(u)]
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.extend(
            [u.tag for u in new_combat_units])
        action.action_raw.unit_command.ability_id = ABILITY.ATTACK.value
        action.action_raw.unit_command.target_world_space_pos.x = rally_pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = rally_pos[1]
        return [action]

    def _is_valid_rally_new_combat_units(self, dc):
        new_combat_units = [u for u in dc.units_of_types(self._combat_types)
                            if dc.is_new_unit(u)]
        if len(new_combat_units) > 0:
            return True
        else:
            return False

    def _rally_idle_combat_units_to_midfield(self, dc):
        rally_pos = (100, 78)
        unrallied_units = [u for u in dc.idle_units_of_types(self._combat_types)
                           if utils.distance(u, rally_pos) > 12]
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.extend(
            [u.tag for u in unrallied_units])
        action.action_raw.unit_command.ability_id = ABILITY.ATTACK.value
        action.action_raw.unit_command.target_world_space_pos.x = rally_pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = rally_pos[1]
        return [action]

    def _is_valid_rally_idle_combat_units_to_midfield(self, dc):
        rally_pos = (100, 78)
        unrallied_units = [u for u in dc.idle_units_of_types(self._combat_types)
                           if utils.distance(u, rally_pos) > 12]
        if len(unrallied_units) > 10:
            return True
        else:
            return False

    def _all_attack_closest_unit(self, dc):
        self._set_attack_status(dc.units_of_types(self._combat_types))
        return []

    def _is_valid_all_attack_closest_unit_30(self, dc):
        if (len(dc.units_of_types(self._combat_types)) > 30 and
            len(dc.units_of_alliance(ALLY_TYPE.ENEMY.value)) > 0):
            return True
        else:
            return False

    def _is_valid_all_attack_closest_unit_20(self, dc):
        if (len(dc.units_of_types(self._combat_types)) > 20 and
            len(dc.units_of_alliance(ALLY_TYPE.ENEMY.value)) > 0):
            return True
        else:
            return False

    def _all_units_of_attack_status_do_micro_attack(self, dc):
        attacking_units = [u for u in dc.units_of_types(self._combat_types)
                           if self._is_attack_status(u)]
        enemy_units = dc.units_of_alliance(ALLY_TYPE.ENEMY.value)
        return self._micro_attack(attacking_units, enemy_units)

    def _is_valid_all_units_of_attack_do_micro_attack(self, dc):
        attacking_units = [u for u in dc.units_of_types(self._combat_types)
                           if self._is_attack_status(u)]
        enemy_units = dc.units_of_alliance(ALLY_TYPE.ENEMY.value)
        if len(attacking_units) > 0 and len(enemy_units) > 0:
            return True
        else:
            return False

    def _micro_attack(self, combat_units, enemy_units):

        def flee_or_fight(unit, target_units):
            assert len(target_units) > 0
            closest_target = utils.closest_unit(unit, target_units)
            closest_dist = utils.closest_distance(unit, enemy_units)
            strongest_health = utils.strongest_health(combat_units)
            if (closest_dist < 5.0 and
                unit.float_attr.health / unit.float_attr.health_max < 0.3 and
                strongest_health > 0.9):
                x = unit.float_attr.pos_x + (unit.float_attr.pos_x - \
                    closest_target.float_attr.pos_x) * 0.2
                y = unit.float_attr.pos_y + (unit.float_attr.pos_y - \
                    closest_target.float_attr.pos_y) * 0.2
                action = sc_pb.Action()
                action.action_raw.unit_command.unit_tags.append(unit.tag)
                # TODO: --> ATTACK_ATTACK ?
                action.action_raw.unit_command.ability_id = ABILITY.MOVE.value
                action.action_raw.unit_command.target_world_space_pos.x = x
                action.action_raw.unit_command.target_world_space_pos.y = y
                return action
            else:
                action = sc_pb.Action()
                action.action_raw.unit_command.unit_tags.append(unit.tag)
                action.action_raw.unit_command.ability_id = \
                    ABILITY.ATTACK_ATTACK.value
                action.action_raw.unit_command.target_unit_tag = \
                    closest_target.tag
                return action

        air_combat_units = [u for u in combat_units
                            if (ATTACK_FORCE[u.unit_type].can_attack_air and
                                not ATTACK_FORCE[u.unit_type].can_attack_land)]
        land_combat_units = [u for u in combat_units
                             if (not ATTACK_FORCE[u.unit_type].can_attack_air and
                                 ATTACK_FORCE[u.unit_type].can_attack_land)]
        air_land_combat_units = [u for u in combat_units
                                 if (ATTACK_FORCE[u.unit_type].can_attack_air and
                                     ATTACK_FORCE[u.unit_type].can_attack_land)]
        air_enemy_units = [u for u in enemy_units if u.bool_attr.is_flying]
        land_enemy_units = [u for u in enemy_units if not u.bool_attr.is_flying]
        actions = []
        for unit in air_combat_units:
            if len(air_enemy_units) > 0:
                actions.append(flee_or_fight(unit, air_enemy_units))
        for unit in land_combat_units:
            if len(land_enemy_units) > 0:
                actions.append(flee_or_fight(unit, land_enemy_units))
        for unit in air_land_combat_units:
            if len(enemy_units) > 0:
                actions.append(
                    flee_or_fight(unit, air_enemy_units + land_enemy_units))
        return actions

    def _set_attack_status(self, units):
        for u in units:
            self._attack_unit_tags.add(u.tag)

    def _is_attack_status(self, unit):
        return unit.tag in self._attack_unit_tags
