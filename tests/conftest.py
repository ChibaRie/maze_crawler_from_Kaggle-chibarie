from types import SimpleNamespace

import pytest


def make_config(**overrides):
    base = dict(
        episodeSteps=501,
        width=20,
        height=20,
        factoryEnergy=1000,
        scoutCost=50,
        workerCost=200,
        minerCost=300,
        scoutMaxEnergy=100,
        workerMaxEnergy=300,
        minerMaxEnergy=500,
        wallBuildCost=100,
        wallRemoveCost=100,
        transformCost=100,
        mineMaxEnergy=1000,
        mineRate=50,
        energyPerTurn=1,
        factoryBuildCooldown=10,
        factoryJumpCooldown=20,
        factoryMovePeriod=2,
        workerMovePeriod=2,
        minerMovePeriod=2,
        visionFactory=4,
        visionScout=5,
        visionWorker=3,
        visionMiner=3,
        scrollStartInterval=4,
        scrollEndInterval=1,
        scrollRampSteps=400,
        crystalDensity=0.06,
        miningNodeDensity=0.03,
        doorProbability=0.08,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_obs(
    *,
    player=0,
    south=0,
    north=19,
    width=20,
    height=20,
    walls=None,
    crystals=None,
    robots=None,
    mines=None,
    miningNodes=None,
):
    if walls is None:
        walls = [0] * (width * height)
    return SimpleNamespace(
        player=player,
        southBound=south,
        northBound=north,
        walls=walls,
        crystals=crystals or {},
        robots=robots or {},
        mines=mines or {},
        miningNodes=miningNodes or {},
    )


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def obs_factory():
    return make_obs


def factory_robot(uid="f0", col=5, row=2, energy=1000, owner=0,
                  move_cd=0, jump_cd=0, build_cd=0):
    return uid, [0, col, row, energy, owner, move_cd, jump_cd, build_cd]


@pytest.fixture
def factory():
    return factory_robot
