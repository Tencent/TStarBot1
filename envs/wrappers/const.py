from enum import Enum, unique

from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE


@unique
class ALLY_TYPE(Enum):
    SELF = 1
    ALLY = 2
    NEUTRAL = 3
    ENEMY = 4


@unique
class PLAYER_FEATURE(Enum):
    PLAYER_ID = 0
    MINERALS = 1
    VESPENE = 2
    FOOD_USED = 3
    FOOD_CAP = 4
    FOOD_ARMY = 5
    FOOD_WORKER = 6
    IDLE_WORKER_COUNT = 7
    ARMY_COUNT = 8
    WARP_GATE_COUNT = 9
    LARVA_COUNT = 10


AREA_COLLISION_BUILDINGS = {
    UNIT_TYPE.ZERG_HATCHERY.value,
    UNIT_TYPE.ZERG_LAIR.value,
    UNIT_TYPE.ZERG_HIVE.value,
    UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
    UNIT_TYPE.ZERG_ROACHWARREN.value,
    UNIT_TYPE.ZERG_HYDRALISKDEN.value,
    UNIT_TYPE.ZERG_EXTRACTOR.value,
    UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
    UNIT_TYPE.NEUTRAL_MINERALFIELD.value,
    UNIT_TYPE.NEUTRAL_MINERALFIELD750.value,
    UNIT_TYPE.NEUTRAL_VESPENEGEYSER.value
}


MAXIMUM_NUM = {
    UNIT_TYPE.ZERG_HATCHERY.value: 6,
    UNIT_TYPE.ZERG_LAIR.value: 6,
    UNIT_TYPE.ZERG_HIVE.value: 6,
    UNIT_TYPE.ZERG_SPAWNINGPOOL.value: 1,
    UNIT_TYPE.ZERG_ROACHWARREN.value: 1,
    UNIT_TYPE.ZERG_HYDRALISKDEN.value: 1,
    UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value: 3
}


class AttackAttr(object):

    def __init__(self, can_attack_land, can_attack_air):
        self.can_attack_land = can_attack_land
        self.can_attack_air = can_attack_air

ATTACK_FORCE = {
    UNIT_TYPE.ZERG_ZERGLING.value: AttackAttr(True, False),
    UNIT_TYPE.ZERG_ROACH.value: AttackAttr(True, False),
    UNIT_TYPE.ZERG_HYDRALISK.value: AttackAttr(True, True),
}
