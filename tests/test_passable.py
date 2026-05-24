import pytest


def test_is_fixed_wall_perimeter(config):
    from main import is_fixed_wall

    # west wall of col 0, east wall of col width-1
    assert is_fixed_wall(config, (0, 5), "WEST")
    assert is_fixed_wall(config, (19, 5), "EAST")
    # interior
    assert not is_fixed_wall(config, (5, 5), "WEST")


def test_is_fixed_wall_mirror_axis(config):
    from main import is_fixed_wall

    # width=20, mirror axis at col 9|10
    assert is_fixed_wall(config, (9, 5), "EAST")
    assert is_fixed_wall(config, (10, 5), "WEST")
    assert not is_fixed_wall(config, (8, 5), "EAST")
    assert not is_fixed_wall(config, (11, 5), "WEST")


def test_passable_blocks_walls(config):
    from main import build_context, memory_update, passable_strict
    from tests.conftest import make_obs, factory_robot
    from tests.test_memory import _fresh_mem

    mem = _fresh_mem()
    walls = [0] * 400
    walls[2 * 20 + 5] = 1  # north wall at (5, 2)
    obs = make_obs(walls=walls, robots=dict([factory_robot(col=5, row=2)]))
    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    assert not passable_strict(ctx, (5, 2), "NORTH")
    assert passable_strict(ctx, (5, 2), "EAST")


def test_passable_strict_rejects_danger_rows(config):
    from main import build_context, memory_update, passable_strict
    from tests.conftest import make_obs, factory_robot
    from tests.test_memory import _fresh_mem

    mem = _fresh_mem()
    obs = make_obs(south=10, robots=dict([factory_robot(col=5, row=13)]))
    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    # row 12 is in danger_rows (south=10, horizon=3 → {10,11,12})
    assert not passable_strict(ctx, (5, 13), "SOUTH")


def test_passable_loose_ignores_danger(config):
    from main import build_context, memory_update, passable_loose
    from tests.conftest import make_obs, factory_robot
    from tests.test_memory import _fresh_mem

    mem = _fresh_mem()
    obs = make_obs(south=10, robots=dict([factory_robot(col=5, row=13)]))
    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    assert passable_loose(ctx, (5, 13), "SOUTH")


def test_passable_blocks_off_board(config):
    from main import build_context, memory_update, passable_strict
    from tests.conftest import make_obs, factory_robot
    from tests.test_memory import _fresh_mem

    mem = _fresh_mem()
    obs = make_obs(south=0, north=19,
                   robots=dict([factory_robot(col=5, row=19)]))
    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    assert not passable_strict(ctx, (5, 19), "NORTH")
    assert not passable_strict(ctx, (0, 5), "WEST")
