from tests.conftest import make_obs, factory_robot


def _fresh_mem():
    from main import _MEM
    _MEM["walls"].clear()
    _MEM["mines"].clear()
    _MEM["mining_nodes"].clear()
    _MEM["roles"].clear()
    _MEM["targets"].clear()
    _MEM["last_actions"].clear()
    _MEM["turn"] = 0
    _MEM["enemy_factory_seen"] = None
    return _MEM


def test_walls_translate_obs_to_global_coords(config):
    from main import memory_update

    mem = _fresh_mem()
    walls = [0] * (20 * 20)
    walls[3 * 20 + 5] = 1  # row offset 3, col 5 → global row south + 3
    obs = make_obs(south=10, north=29, walls=walls)

    memory_update(obs, config, mem)

    assert mem["walls"][(5, 13)] == 1


def test_walls_skip_undiscovered_cells(config):
    from main import memory_update

    mem = _fresh_mem()
    walls = [-1] * (20 * 20)
    walls[2 * 20 + 4] = 5
    obs = make_obs(south=0, walls=walls)

    memory_update(obs, config, mem)

    assert (4, 2) in mem["walls"]
    assert (3, 2) not in mem["walls"]


def test_walls_below_south_are_pruned(config):
    from main import memory_update

    mem = _fresh_mem()
    mem["walls"][(5, 3)] = 1
    mem["walls"][(5, 12)] = 1
    obs = make_obs(south=10, walls=[0] * 400)

    memory_update(obs, config, mem)

    assert (5, 3) not in mem["walls"]
    assert (5, 12) in mem["walls"]


def test_roles_pruned_for_dead_units(config):
    from main import memory_update

    mem = _fresh_mem()
    mem["roles"]["dead"] = "EXPLORER"
    mem["roles"]["alive"] = "GUARD"
    me_uid, me = factory_robot(uid="alive", owner=0)
    obs = make_obs(robots={me_uid: me})

    memory_update(obs, config, mem)

    assert "dead" not in mem["roles"]
    assert "alive" in mem["roles"]


def test_turn_counter_increments(config):
    from main import memory_update

    mem = _fresh_mem()
    obs = make_obs()
    memory_update(obs, config, mem)
    memory_update(obs, config, mem)

    assert mem["turn"] == 2
