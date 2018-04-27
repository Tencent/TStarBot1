import gym
import platform
import numpy as np

from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import UPGRADE_ID as UPGRADE
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb

from envs.space import PySC2RawObservation
from envs.space import MaskableDiscrete
from envs.common.data_context import DataContext
from envs.actions.function import Function
from envs.actions.produce import ProduceActions
from envs.actions.build import BuildActions
from envs.actions.upgrade import UpgradeActions
from envs.actions.morph import MorphActions
from envs.actions.resource import ResourceActions
from envs.actions.combat import CombatActions


class ZergActionWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ZergActionWrapper, self).__init__(env)
        assert isinstance(env.observation_space, PySC2RawObservation)

        self._dc = DataContext()
        self._produce_mgr = ProduceActions()
        self._build_mgr = BuildActions()
        self._upgrade_mgr = UpgradeActions()
        self._morph_mgr = MorphActions()
        self._resource_mgr = ResourceActions()
        self._combat_mgr = CombatActions()

        self._actions = [
            self._action_do_nothing(),
            self._build_mgr.action("build_extractor", UNIT_TYPE.ZERG_EXTRACTOR.value),
            self._build_mgr.action("build_spawning_pool", UNIT_TYPE.ZERG_SPAWNINGPOOL.value),
            self._build_mgr.action("build_roach_warren", UNIT_TYPE.ZERG_ROACHWARREN.value),
            self._build_mgr.action("build_hydraliskden", UNIT_TYPE.ZERG_HYDRALISKDEN.value),
            self._build_mgr.action("build_hatchery", UNIT_TYPE.ZERG_HATCHERY.value),
            self._build_mgr.action("build_evolution_chamber", UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value),
            self._build_mgr.action("build_baneling_nest", UNIT_TYPE.ZERG_BANELINGNEST.value),
            self._build_mgr.action("build_infestation_pit", UNIT_TYPE.ZERG_INFESTATIONPIT.value),
            self._build_mgr.action("build_spire", UNIT_TYPE.ZERG_SPIRE.value),
            self._build_mgr.action("build_ultralisk_cavern", UNIT_TYPE.ZERG_ULTRALISKCAVERN.value),
            self._build_mgr.action("build_nydus_network", UNIT_TYPE.ZERG_NYDUSNETWORK.value),
            self._build_mgr.action("build_spine_crawler", UNIT_TYPE.ZERG_SPINECRAWLER.value),
            self._build_mgr.action("build_spore_crawler", UNIT_TYPE.ZERG_SPORECRAWLER.value),
            self._build_mgr.action("build_lurker_den", UNIT_TYPE.ZERG_LURKERDENMP.value),
            self._morph_mgr.action("morph_lair", UNIT_TYPE.ZERG_LAIR.value),
            self._morph_mgr.action("morph_hive", UNIT_TYPE.ZERG_HIVE.value),
            self._morph_mgr.action("morph_greater_spire", UNIT_TYPE.ZERG_GREATERSPIRE.value),
            self._produce_mgr.action("produce_drone", UNIT_TYPE.ZERG_DRONE.value),
            self._produce_mgr.action("produce_zergling", UNIT_TYPE.ZERG_ZERGLING.value),
            self._morph_mgr.action("morph_baneling", UNIT_TYPE.ZERG_BANELING.value),
            self._produce_mgr.action("produce_roach", UNIT_TYPE.ZERG_ROACH.value),
            self._morph_mgr.action("morph_ravager", UNIT_TYPE.ZERG_RAVAGER.value),
            self._produce_mgr.action("produce_hydralisk", UNIT_TYPE.ZERG_HYDRALISK.value),
            self._morph_mgr.action("morph_lurker", UNIT_TYPE.ZERG_LURKERMP.value),
            self._produce_mgr.action("produce_viper", UNIT_TYPE.ZERG_VIPER.value),
            self._produce_mgr.action("produce_mutalisk", UNIT_TYPE.ZERG_MUTALISK.value),
            self._produce_mgr.action("produce_corruptor", UNIT_TYPE.ZERG_CORRUPTOR.value),
            self._morph_mgr.action("morph_broodlord", UNIT_TYPE.ZERG_BROODLORD.value),
            self._produce_mgr.action("produce_swarmhost", UNIT_TYPE.ZERG_SWARMHOSTMP.value),
            self._produce_mgr.action("produce_infestor", UNIT_TYPE.ZERG_INFESTOR.value),
            self._produce_mgr.action("produce_ultralisk", UNIT_TYPE.ZERG_ULTRALISK.value),
            self._produce_mgr.action("produce_overlord", UNIT_TYPE.ZERG_OVERLORD.value),
            self._morph_mgr.action("morph_overseer", UNIT_TYPE.ZERG_OVERSEER.value),
            self._produce_mgr.action("produce_queen", UNIT_TYPE.ZERG_QUEEN.value),
            self._produce_mgr.action("produce_nydus_worm", UNIT_TYPE.ZERG_NYDUSCANAL.value),
            self._upgrade_mgr.action("upgrade_burrow", UPGRADE.BURROW.value),
            self._upgrade_mgr.action("upgrade_centrifical_hooks", UPGRADE.CENTRIFICALHOOKS.value),
            self._upgrade_mgr.action("upgrade_chitions_plating", UPGRADE.CHITINOUSPLATING.value),
            self._upgrade_mgr.action("upgrade_evolve_grooved_spines", UPGRADE.EVOLVEGROOVEDSPINES.value),
            self._upgrade_mgr.action("upgrade_evolve_muscular_augments", UPGRADE.EVOLVEMUSCULARAUGMENTS.value),
            self._upgrade_mgr.action("upgrade_gliare_constitution", UPGRADE.GLIALRECONSTITUTION.value),
            self._upgrade_mgr.action("upgrade_infestor_energy_upgrade", UPGRADE.INFESTORENERGYUPGRADE.value),
            self._upgrade_mgr.action("upgrade_neural_parasite", UPGRADE.NEURALPARASITE.value),
            self._upgrade_mgr.action("upgrade_overlord_speed", UPGRADE.OVERLORDSPEED.value),
            self._upgrade_mgr.action("upgrade_tunneling_claws", UPGRADE.TUNNELINGCLAWS.value),
            self._upgrade_mgr.action("upgrade_flyer_armors_level1", UPGRADE.ZERGFLYERARMORSLEVEL1.value),
            self._upgrade_mgr.action("upgrade_flyer_armors_level2", UPGRADE.ZERGFLYERARMORSLEVEL2.value),
            self._upgrade_mgr.action("upgrade_flyer_armors_level3", UPGRADE.ZERGFLYERARMORSLEVEL3.value),
            self._upgrade_mgr.action("upgrade_flyer_weapons_level1", UPGRADE.ZERGFLYERWEAPONSLEVEL1.value),
            self._upgrade_mgr.action("upgrade_flyer_weapons_level2", UPGRADE.ZERGFLYERWEAPONSLEVEL2.value),
            self._upgrade_mgr.action("upgrade_flyer_weapons_level3", UPGRADE.ZERGFLYERWEAPONSLEVEL3.value),
            self._upgrade_mgr.action("upgrade_ground_armors_level1", UPGRADE.ZERGGROUNDARMORSLEVEL1.value),
            self._upgrade_mgr.action("upgrade_ground_armors_level2", UPGRADE.ZERGGROUNDARMORSLEVEL2.value),
            self._upgrade_mgr.action("upgrade_ground_armors_level3", UPGRADE.ZERGGROUNDARMORSLEVEL3.value),
            self._upgrade_mgr.action("upgrade_zergling_attack_speed", UPGRADE.ZERGLINGATTACKSPEED.value),
            self._upgrade_mgr.action("upgrade_zergling_moving_speed", UPGRADE.ZERGLINGMOVEMENTSPEED.value),
            self._upgrade_mgr.action("upgrade_melee_weapons_level1", UPGRADE.ZERGMELEEWEAPONSLEVEL1.value),
            self._upgrade_mgr.action("upgrade_melee_weapons_level2", UPGRADE.ZERGMELEEWEAPONSLEVEL2.value),
            self._upgrade_mgr.action("upgrade_melee_weapons_level3", UPGRADE.ZERGMELEEWEAPONSLEVEL3.value),
            self._upgrade_mgr.action("upgrade_missile_weapons_level1", UPGRADE.ZERGMISSILEWEAPONSLEVEL1.value),
            self._upgrade_mgr.action("upgrade_missile_weapons_level2", UPGRADE.ZERGMISSILEWEAPONSLEVEL2.value),
            self._upgrade_mgr.action("upgrade_missile_weapons_level3", UPGRADE.ZERGMISSILEWEAPONSLEVEL3.value),
            self._resource_mgr.action_queens_inject_larva,
            self._resource_mgr.action_some_workers_gather_gas,
            self._combat_mgr.action_rally_idle_combat_units_to_midfield,
            self._combat_mgr.action_all_attack_30,
            self._combat_mgr.action_all_attack_20
            # ZERG_LOCUST, ZERG_CHANGELING not included
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

    @property
    def action_names(self):
        return [action.name for action in self._actions]

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
