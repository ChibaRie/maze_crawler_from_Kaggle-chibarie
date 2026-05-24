def _trivial_ctx(config, walls=None, south=0, north=19, factory_at=(5, 2)):
    from main import build_context, memory_update
    from tests.conftest import make_obs, factory_robot
    from tests.test_memory import _fresh_mem

    mem = _fresh_mem()
    if walls is None:
        walls = [0] * 400
    obs = make_obs(
        south=south, north=north, walls=walls,
        robots=dict([factory_robot(col=factory_at[0], row=factory_at[1])]),
    )
    memory_update(obs, config, mem)
    return build_context(obs, config, mem)


def test_bfs_returns_idle_at_goal(config):
    from main import bfs, passable_loose

    ctx = _trivial_ctx(config)
    result = bfs((5, 2), lambda c: c == (5, 2),
                 passable_fn=lambda cell, d: passable_loose(ctx, cell, d))
    assert result == (0, "IDLE")


def test_bfs_finds_path_open_field(config):
    from main import bfs, passable_loose

    ctx = _trivial_ctx(config, factory_at=(5, 2))
    result = bfs((5, 2), lambda c: c == (5, 5),
                 passable_fn=lambda cell, d: passable_loose(ctx, cell, d))
    assert result == (3, "NORTH")


def test_bfs_returns_none_when_blocked(config):
    from main import bfs, passable_loose

    walls = [0] * 400
    # box (5, 2) in with walls on all 4 sides
    walls[2 * 20 + 5] = 1 | 2 | 4 | 8
    ctx = _trivial_ctx(config, walls=walls)
    result = bfs((5, 2), lambda c: c == (5, 5),
                 passable_fn=lambda cell, d: passable_loose(ctx, cell, d))
    assert result is None


def test_bfs_first_step_avoids_occupied(config):
    from main import bfs, passable_loose

    ctx = _trivial_ctx(config, factory_at=(5, 2))
    # Block the direct NORTH path at (5,3) with occupied set
    result = bfs(
        (5, 2),
        lambda c: c == (5, 5),
        passable_fn=lambda cell, d: passable_loose(ctx, cell, d),
        occupied=frozenset({(5, 3)}),
    )
    # Must detour east or west
    assert result is not None
    assert result[1] in {"EAST", "WEST"}


def test_bfs_respects_max_dist(config):
    from main import bfs, passable_loose

    ctx = _trivial_ctx(config, factory_at=(5, 2))
    result = bfs((5, 2), lambda c: c == (5, 19),
                 passable_fn=lambda cell, d: passable_loose(ctx, cell, d),
                 max_dist=2)
    assert result is None
