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


def test_context_exposes_my_units_and_factory(config):
    from main import memory_update, build_context

    mem = _fresh_mem()
    me_f_uid, me_f = factory_robot(uid="f", col=5, row=3, owner=0)
    me_s_uid, me_s = "s1", [1, 6, 4, 100, 0, 0, 0, 0]
    enemy_uid, enemy = factory_robot(uid="ef", col=14, row=3, owner=1)
    obs = make_obs(robots={me_f_uid: me_f, me_s_uid: me_s, enemy_uid: enemy})

    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    assert ctx.my_factory.uid == "f"
    assert {u.uid for u in ctx.my_units} == {"f", "s1"}
    assert {u.uid for u in ctx.enemy_units} == {"ef"}
    assert ctx.enemy_factory.uid == "ef"


def test_context_walls_merge_obs_and_memory(config):
    from main import memory_update, build_context

    mem = _fresh_mem()
    mem["walls"][(5, 8)] = 1   # remembered, currently outside obs window
    walls = [0] * 400
    walls[2 * 20 + 4] = 2
    obs = make_obs(south=10, walls=walls)

    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    assert ctx.walls.get((4, 12)) == 2
    # remembered cell at row 8 is below south=10 → pruned by memory_update
    assert (5, 8) not in ctx.walls


def test_context_danger_rows(config):
    from main import memory_update, build_context

    mem = _fresh_mem()
    obs = make_obs(south=10)
    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    # SAFETY_HORIZON = 3 → rows 10, 11, 12 all flagged as danger
    assert 10 in ctx.danger_rows
    assert 12 in ctx.danger_rows
    assert 13 not in ctx.danger_rows
