from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation
from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.envs.observations.spatial_features import UnitTypeCountMapFeature
from sc2learner.envs.observations.spatial_features import AllianceCountMapFeature
from sc2learner.envs.observations.nonspatial_features import PlayerFeature
from sc2learner.envs.observations.nonspatial_features import UnitTypeCountFeature
from sc2learner.envs.observations.nonspatial_features import UnitStatCountFeature
from sc2learner.envs.observations.nonspatial_features import GameProgressFeature
from sc2learner.envs.observations.nonspatial_features import ActionSeqFeature


class ZergNonspatialObservationAgentWrapper(object):

  def __init__(self, agent, action_space):
    self._unit_count_feature = UnitTypeCountFeature(
        type_list=[UNIT_TYPE.ZERG_LARVA.value,
                   UNIT_TYPE.ZERG_DRONE.value,
                   UNIT_TYPE.ZERG_ZERGLING.value,
                   UNIT_TYPE.ZERG_BANELING.value,
                   UNIT_TYPE.ZERG_ROACH.value,
                   UNIT_TYPE.ZERG_ROACHBURROWED.value,
                   UNIT_TYPE.ZERG_RAVAGER.value,
                   UNIT_TYPE.ZERG_HYDRALISK.value,
                   UNIT_TYPE.ZERG_LURKERMP.value,
                   UNIT_TYPE.ZERG_LURKERMPBURROWED.value,
                   UNIT_TYPE.ZERG_VIPER.value,
                   UNIT_TYPE.ZERG_MUTALISK.value,
                   UNIT_TYPE.ZERG_CORRUPTOR.value,
                   UNIT_TYPE.ZERG_BROODLORD.value,
                   UNIT_TYPE.ZERG_SWARMHOSTMP.value,
                   UNIT_TYPE.ZERG_LOCUSTMP.value,
                   UNIT_TYPE.ZERG_INFESTOR.value,
                   UNIT_TYPE.ZERG_ULTRALISK.value,
                   UNIT_TYPE.ZERG_BROODLING.value,
                   UNIT_TYPE.ZERG_OVERLORD.value,
                   UNIT_TYPE.ZERG_OVERSEER.value,
                   UNIT_TYPE.ZERG_QUEEN.value,
                   UNIT_TYPE.ZERG_CHANGELING.value,
                   UNIT_TYPE.ZERG_SPINECRAWLER.value,
                   UNIT_TYPE.ZERG_SPORECRAWLER.value,
                   UNIT_TYPE.ZERG_NYDUSCANAL.value,
                   UNIT_TYPE.ZERG_EXTRACTOR.value,
                   UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                   UNIT_TYPE.ZERG_ROACHWARREN.value,
                   UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                   UNIT_TYPE.ZERG_HATCHERY.value,
                   UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                   UNIT_TYPE.ZERG_BANELINGNEST.value,
                   UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                   UNIT_TYPE.ZERG_SPIRE.value,
                   UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                   UNIT_TYPE.ZERG_NYDUSNETWORK.value,
                   UNIT_TYPE.ZERG_LURKERDENMP.value,
                   UNIT_TYPE.ZERG_LAIR.value,
                   UNIT_TYPE.ZERG_HIVE.value,
                   UNIT_TYPE.ZERG_GREATERSPIRE.value])
    self._unit_stat_count_feature = UnitStatCountFeature()
    self._player_feature = PlayerFeature()
    self._game_progress_feature = GameProgressFeature()
    self._action_seq_feature = ActionSeqFeature(action_space.n, 10)

    n_dims = sum([self._unit_stat_count_feature.num_dims,
                  self._unit_count_feature.num_dims,
                  self._player_feature.num_dims,
                  self._game_progress_feature.num_dims,
                  self._action_seq_feature.num_dims])
    self.observation_space = spaces.Box(0.0, float('inf'), [n_dims])
    self._agent = agent

  def act(self, observation):
    action = self._agent.act(self._observation(observation))
    self._action_seq_feature.push_action(action)
    return action

  def reset(self, observation):
    self._agent.reset(observation)
    self._action_seq_feature.reset()

  def _observation(self, observation):
    unit_type_feat = self._unit_count_feature.features(observation)
    unit_stat_feat = self._unit_stat_count_feature.features(observation)
    player_feat = self._player_feature.features(observation)
    game_progress_feat = self._game_progress_feature.features(observation)
    action_seq_feat = self._action_seq_feature.features()

    nonspatial_feat = np.concatenate([unit_type_feat,
                                      unit_stat_feat,
                                      player_feat,
                                      game_progress_feat,
                                      action_seq_feat])

    return nonspatial_feat
