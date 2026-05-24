from tests.conftest import make_obs, factory_robot
from tests.test_memory import _fresh_mem


def _ctx_with_units(config, my, enemies=(), walls=None, south=0, north=19):
    from main import build_context, memory_update

    mem = _fresh_mem()
    robots = {}
    for u in my:
        robots[u[0]] = u[1]
    for u in enemies:
        robots[u[0]] = u[1]
    if walls is None:
        walls = [0] * 400
    obs = make_obs(walls=walls, south=south, north=north, robots=robots)
    memory_update(obs, config, mem)
    return build_context(obs, config, mem)


def test_predict_next_cell_movement(config):
    from main import predict_next_cell, _parse_unit

    unit = _parse_unit("u", [1, 5, 5, 100, 0, 0, 0, 0])
    assert predict_next_cell(unit, "NORTH") == (5, 6)
    assert predict_next_cell(unit, "IDLE") == (5, 5)
    assert predict_next_cell(unit, "TRANSFORM") == (5, 5)


def test_predict_next_cell_jump(config):
    from main import predict_next_cell, _parse_unit

    fac = _parse_unit("f", [0, 5, 5, 1000, 0, 0, 0, 0])
    assert predict_next_cell(fac, "JUMP_NORTH") == (5, 7)
    assert predict_next_cell(fac, "JUMP_EAST") == (7, 5)


def test_predict_next_cell_build_factory_uses_north_spawn(config):
    from main import predict_next_cell, _parse_unit

    fac = _parse_unit("f", [0, 5, 5, 1000, 0, 0, 0, 0])
    assert predict_next_cell(fac, "BUILD_SCOUT") == (5, 6)


def test_death_filter_drops_off_board(config):
    from main import death_filter, _parse_unit

    ctx = _ctx_with_units(
        config,
        my=[factory_robot(uid="me", col=5, row=19, owner=0)],
        north=19,
    )
    me = ctx.my_factory
    candidates = ["NORTH", "EAST", "WEST", "SOUTH", "IDLE"]
    filtered = death_filter(ctx, me, candidates, reservations=set())
    assert "NORTH" not in filtered  # would walk off the north edge
    assert "IDLE" in filtered


def test_death_filter_drops_higher_crush_friend(config):
    from main import death_filter, _parse_unit

    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    # a friendly worker (higher crush than scout) at (5,3)
    worker = ("w", [2, 5, 3, 200, 0, 0, 0, 0])
    scout = ("s", [1, 5, 2, 100, 0, 0, 0, 0])
    ctx = _ctx_with_units(config, my=[me_fac, worker, scout])
    scout_unit = next(u for u in ctx.my_units if u.uid == "s")
    reservations = {(5, 3)}  # worker reserves its cell
    filtered = death_filter(ctx, scout_unit, ["NORTH", "IDLE"], reservations,
                            reservation_types={(5, 3): 2})
    assert "NORTH" not in filtered


def test_direction_score_prefers_north(config):
    from main import direction_score

    assert direction_score("NORTH") > direction_score("EAST")
    assert direction_score("EAST") > direction_score("SOUTH")
    assert direction_score("WEST") == direction_score("EAST")


def test_assign_roles_worker_with_wall_target(config):
    from main import assign_roles

    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    worker = ("w", [2, 5, 3, 300, 0, 0, 0, 0])
    walls = [0] * 400
    walls[5 * 20 + 5] = 1  # wall north of (5,5) — bottleneck on the way north
    ctx = _ctx_with_units(config, my=[me_fac, worker], walls=walls)

    roles = assign_roles(ctx)
    assert roles["f"] == "FACTORY"
    # With a wall on the route, the worker is a SAPPER candidate; either way,
    # the role must be one of the legal worker roles.
    assert roles["w"] in {"SAPPER", "GUARD"}


def test_assign_roles_sticky_per_period(config):
    from main import assign_roles

    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    scout = ("s", [1, 5, 3, 100, 0, 0, 0, 0])
    ctx = _ctx_with_units(config, my=[me_fac, scout])
    ctx.mem["roles"]["s"] = "EXPLORER"
    ctx.mem["turn"] = 5  # not a re-assess turn
    roles = assign_roles(ctx)
    assert roles["s"] == "EXPLORER"


def test_frontier_score_prefers_more_unknown(config):
    from main import frontier_score, build_context, memory_update

    mem = _fresh_mem()
    walls = [-1] * 400
    walls[2 * 20 + 5] = 0  # cell (5, 2) known
    walls[5 * 20 + 5] = 0  # cell (5, 5) known
    obs = make_obs(walls=walls, robots=dict([factory_robot(col=5, row=2)]))
    memory_update(obs, config, mem)
    ctx = build_context(obs, config, mem)

    s_low = frontier_score(ctx, (5, 2))
    s_high = frontier_score(ctx, (5, 5))
    # both are surrounded by unknown; north-bias makes (5,5) higher
    assert s_high >= s_low


def test_pick_factory_build_first_is_scout(config):
    from main import pick_factory_build

    me_fac = factory_robot(uid="f", col=5, row=2, energy=1000, owner=0)
    ctx = _ctx_with_units(config, my=[me_fac])
    action = pick_factory_build(ctx, ctx.my_factory)
    assert action == "BUILD_SCOUT"


def test_pick_factory_build_low_energy_skips(config):
    from main import pick_factory_build

    me_fac = factory_robot(uid="f", col=5, row=2,
                           energy=int(1000 * 0.2), owner=0)
    ctx = _ctx_with_units(config, my=[me_fac])
    action = pick_factory_build(ctx, ctx.my_factory)
    assert action is None


def test_pick_factory_build_late_game_stops(config):
    from main import pick_factory_build

    me_fac = factory_robot(uid="f", col=5, row=2, energy=1000, owner=0)
    ctx = _ctx_with_units(config, my=[me_fac])
    # Late-game stop kicks in when episodeSteps - turn < LATE_GAME_STOP_BUILD
    ctx.mem["turn"] = config.episodeSteps - 5
    # Rebuild context so ctx.turn reflects the new turn count.
    from main import build_context
    ctx = build_context(ctx.obs, config, ctx.mem)
    action = pick_factory_build(ctx, ctx.my_factory)
    assert action is None


def test_assign_targets_explorer_picks_frontier(config):
    from main import assign_targets, assign_roles

    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    scout = ("s", [1, 5, 3, 100, 0, 0, 0, 0])
    walls = [-1] * 400
    walls[3 * 20 + 5] = 0  # known: scout's cell
    walls[2 * 20 + 5] = 0  # known: factory's cell
    ctx = _ctx_with_units(config, my=[me_fac, scout], walls=walls)
    assign_roles(ctx)
    targets = assign_targets(ctx)
    assert "s" in targets


def test_decide_factory_low_energy_no_build(config):
    from main import decide_unit, assign_roles, assign_targets

    me_fac = factory_robot(uid="f", col=5, row=2,
                           energy=int(1000 * 0.1), owner=0)
    ctx = _ctx_with_units(config, my=[me_fac])
    assign_roles(ctx)
    assign_targets(ctx)
    action = decide_unit(ctx, ctx.my_factory, set(), {})
    assert action != "BUILD_SCOUT"
    assert action in {"NORTH", "IDLE"}


def test_decide_explorer_returns_legal_move(config):
    from main import decide_unit, assign_roles, assign_targets

    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    scout = ("s", [1, 5, 3, 100, 0, 0, 0, 0])
    ctx = _ctx_with_units(config, my=[me_fac, scout])
    assign_roles(ctx)
    assign_targets(ctx)
    scout_unit = next(u for u in ctx.my_units if u.uid == "s")
    action = decide_unit(ctx, scout_unit, set(), {})
    assert action in {"NORTH", "EAST", "SOUTH", "WEST", "IDLE"}


def test_decide_sapper_at_wall_removes(config):
    from main import decide_unit, assign_roles, assign_targets

    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    worker = ("w", [2, 5, 3, 300, 0, 0, 0, 0])
    walls = [0] * 400
    walls[3 * 20 + 5] = 1  # north wall at (5,3)
    walls[4 * 20 + 5] = 4  # mirror south wall at (5,4) — engine consistency, optional
    ctx = _ctx_with_units(config, my=[me_fac, worker], walls=walls)
    # Force SAPPER role
    ctx.mem["roles"]["w"] = "SAPPER"
    ctx.mem["targets"]["w"] = (5, 3)
    worker_unit = next(u for u in ctx.my_units if u.uid == "w")
    action = decide_unit(ctx, worker_unit, set(), {})
    assert action == "REMOVE_NORTH"


def test_decide_sapper_skips_fixed_wall(config):
    from main import decide_unit, assign_roles

    # Worker sitting next to perimeter wall; if assigned a fixed wall target,
    # decide_unit must NOT issue REMOVE.
    me_fac = factory_robot(uid="f", col=5, row=2, owner=0)
    worker = ("w", [2, 0, 3, 300, 0, 0, 0, 0])
    ctx = _ctx_with_units(config, my=[me_fac, worker])
    ctx.mem["roles"]["w"] = "SAPPER"
    ctx.mem["targets"]["w"] = (0, 3)  # would imply REMOVE_WEST → fixed
    worker_unit = next(u for u in ctx.my_units if u.uid == "w")
    action = decide_unit(ctx, worker_unit, set(), {})
    assert not action.startswith("REMOVE_") and not action.startswith("BUILD_")


def test_agent_pre_reserves_factory_escort_cell(config):
    from main import agent

    me_fac_uid, me_fac = factory_robot(uid="f", col=5, row=2,
                                       energy=1000, owner=0)
    # A friendly worker east-adjacent to the escort cell (5, 3).
    worker_uid, worker = "w", [2, 4, 3, 300, 0, 0, 0, 0]
    obs = make_obs(robots={me_fac_uid: me_fac, worker_uid: worker})

    actions = agent(obs, config)

    # Worker must not step EAST onto (5, 3) since it's the factory's escort cell.
    assert actions["w"] != "EAST"
