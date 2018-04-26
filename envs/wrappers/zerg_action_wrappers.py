import gym
import platform
import numpy as np

from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import UPGRADE_ID as UPGRADE
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb

from envs.space import PySC2RawObservation
from envs.space import MaskableDiscrete
from envs.wrappers.data_context import DataContext
from envs.wrappers.produce_mgr import ProduceManager
from envs.wrappers.build_mgr import BuildManager
from envs.wrappers.upgrade_mgr import UpgradeManager
from envs.wrappers.morph_mgr import MorphManager
from envs.wrappers.resource_mgr import ResourceManager
from envs.wrappers.combat_mgr import CombatManager
from envs.wrappers.utils import Function


class ZergActionWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ZergActionWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)

        self._dc = DataContext()
        self._produce_mgr = ProduceManager()
        self._build_mgr = BuildManager()
        self._upgrade_mgr = UpgradeManager()
        self._morph_mgr = MorphManager()
        self._resource_mgr = ResourceManager()
        self._combat_mgr = CombatManager()

        self._actions = [
            self._action_do_nothing(),
            self._build_mgr.action(
                "build_extractor", UNIT_TYPE.ZERG_EXTRACTOR.value),
            self._build_mgr.action(
                "build_spawning_pool", UNIT_TYPE.ZERG_SPAWNINGPOOL.value),
            self._build_mgr.action(
                "build_roach_warren", UNIT_TYPE.ZERG_ROACHWARREN.value),
            self._build_mgr.action(
                "build_hydraliskden", UNIT_TYPE.ZERG_HYDRALISKDEN.value),
            self._build_mgr.action(
                "build_hatchery", UNIT_TYPE.ZERG_HATCHERY.value),
            self._produce_mgr.action(
                "produce_overlord", UNIT_TYPE.ZERG_OVERLORD.value),
            self._produce_mgr.action(
                "produce_drone", UNIT_TYPE.ZERG_DRONE.value),
            self._produce_mgr.action(
                "produce_zergling", UNIT_TYPE.ZERG_ZERGLING.value),
            self._produce_mgr.action(
                "produce_roach", UNIT_TYPE.ZERG_ROACH.value),
            self._produce_mgr.action(
                "produce_queen", UNIT_TYPE.ZERG_QUEEN.value),
            self._produce_mgr.action(
                "produce_hydralisk", UNIT_TYPE.ZERG_HYDRALISK.value),
            self._resource_mgr.action_queens_inject_larva,
            self._resource_mgr.action_some_workers_gather_gas,
            self._morph_mgr.action(
                "morph_lair", UNIT_TYPE.ZERG_LAIR.value),
            self._combat_mgr.action_rally_idle_combat_units_to_midfield,
            self._combat_mgr.action_all_attack_30,
            self._combat_mgr.action_all_attack_20,
            self._build_mgr.action(
                "build_evolution_chamber", UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value),
            self._upgrade_mgr.action(
                "upgrade_melee_attack_1", UPGRADE.ZERGMELEEWEAPONSLEVEL1.value),
            self._upgrade_mgr.action(
                "upgrade_melee_attack_2", UPGRADE.ZERGMELEEWEAPONSLEVEL2.value),
            self._upgrade_mgr.action(
                "upgrade_missile_attack_1", UPGRADE.ZERGMISSILEWEAPONSLEVEL1.value),
            self._upgrade_mgr.action(
                "upgrade_missile_attack_2", UPGRADE.ZERGMISSILEWEAPONSLEVEL2.value),
            self._upgrade_mgr.action(
                "upgrade_ground_armor_1", UPGRADE.ZERGGROUNDARMORSLEVEL1.value),
            self._upgrade_mgr.action(
                "upgrade_ground_armor_2", UPGRADE.ZERGGROUNDARMORSLEVEL2.value)
        ]
        self.action_space = MaskableDiscrete(len(self._actions))

    def step(self, action):
        assert self._action_mask[action] == 1
        actions = self._actions[action].function(self._dc)
        actions_before, actions_after = self._extended_actions()
        final_actions = actions_before + actions + actions_after
        observation, reward, done, info = self.env.step(final_actions)
        self._dc.update(observation)
        self._action_mask = self._get_valid_action_mask()
        return (observation, self._action_mask), reward, done, info

    def reset(self):
        self._combat_mgr.reset()
        observation = self.env.reset()
        self._dc.reset(observation)
        self._action_mask = self._get_valid_action_mask()
        return (observation, self._action_mask)

    def print_actions(self):
        for action_id, action in enumerate(self._actions):
            print("Action ID: %d	Action Name: %s" % (action_id, action.name))

    @property
    def player_position(self):
        if self._dc.init_base_pos[0] < 100:
            return 0
        else:
            return 1

    def _extended_actions(self):
        actions_before = []
        # TODO(@xinghai) : remove this hack
        has_any_unit_selected = any([u.bool_attr.is_selected
                                     for u in self._dc.units])
        if platform.system() != 'Linux' and not has_any_unit_selected:
            actions_before.extend(self._action_select_units_for_mac())
        fn = self._resource_mgr.action_idle_workers_gather_minerals
        if fn.is_valid(self._dc):
            actions_before.extend(fn.function(self._dc))

        actions_after = []
        fn = self._combat_mgr.action_rally_new_combat_units
        if fn.is_valid(self._dc):
            actions_after.extend(fn.function(self._dc))
        fn = self._combat_mgr.action_universal_micro_attack
        if fn.is_valid(self._dc):
            actions_after.extend(fn.function(self._dc))

        return actions_before, actions_after

    def _get_valid_action_mask(self):
        ids = [i for i, action in enumerate(self._actions)
               if action.is_valid(self._dc)]
        mask = np.zeros(self.action_space.n)
        mask[ids] = 1
        return mask

    def _action_do_nothing(self):
        return Function(name='do_nothing',
                        function=lambda dc: [],
                        is_valid=lambda dc: True)

    # TODO(@xinghai) : remove this hack
    def _action_select_units_for_mac(self):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = 0
        select = action.action_feature_layer.unit_selection_rect
        out_rect = select.selection_screen_coord.add()
        point.Point(0, 0).assign_to(out_rect.p0)
        point.Point(32, 32).assign_to(out_rect.p1)
        select.selection_add = True
        return [action]
