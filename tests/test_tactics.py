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
