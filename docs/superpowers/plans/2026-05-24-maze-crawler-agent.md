# Maze Crawler Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a competitive single-file `main.py` agent for the Kaggle `maze-crawler` competition using heuristic FSM + BFS pathfinding + light risk evaluation.

**Architecture:** Four layers — memory (module-level `_MEM` dict), context (per-turn read-only view), strategy (role + build decisions), tactics (per-unit action selection). Single file for submission; tests live separately under `tests/`.

**Tech Stack:** Python 3.11+, `kaggle-environments` (already specified in README), `pytest` for tests.

**Reference spec:** `docs/superpowers/specs/2026-05-24-maze-crawler-agent-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `main.py` | Single-file submission. Contains all 4 layers (memory, context, strategy, tactics) + tunables block + `agent` entry point. |
| `tests/__init__.py` | Empty — marks tests as a package. |
| `tests/test_memory.py` | Memory layer: pruning, wall merge, role cleanup. |
| `tests/test_bfs.py` | BFS unit tests: path found, blocked, max_dist cap. |
| `tests/test_tactics.py` | Pipeline smoke tests: legal action returned, death filter respected, fixed wall guard. |
| `tests/test_smoke.py` | End-to-end: build a fake `obs`, call `agent`, assert dict shape. |
| `tests/conftest.py` | Shared fixtures: synthetic `obs` builder, default `config` namespace. |
| `tests/match_random.py` | Local match script — runs vs `random` on 5 seeds, asserts win rate ≥ 80%. Manual run, not pytest. |

`main.py` is the only artifact submitted. Tests are kept out of the tarball.

---

## Tunables (referenced throughout)

The first task installs this constants block at the top of `main.py`. Later tasks reference these names directly.

```python
SAFETY_MARGIN          = 4
SAFETY_HORIZON         = 3
LOW_ENERGY_RATIO       = 0.30
LATE_GAME_STOP_BUILD   = 30
LATE_GAME_HOARD        = 50
EXPLORE_KERNEL         = 5
NORTH_BIAS             = 1.5
WALL_DETOUR_THRESHOLD  = 6
ROLE_REASSIGN_PERIOD   = 8
TRANSFER_OVERFLOW_GAP  = 5
CRYSTAL_DETOUR_BUDGET  = 2
ENEMY_NEAR_RADIUS      = 2
MINER_REACH_LIMIT      = 25
BFS_MAX_DIST           = 80
```

---

## Task 1: Skeleton + Tunables + Backwards-compatible Agent

**Files:**
- Modify: `main.py` (full rewrite of the existing 38-line starter)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/__init__.py` empty.

Create `tests/conftest.py`:

```python
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
```

Create `tests/test_smoke.py`:

```python
from tests.conftest import make_obs, factory_robot


def test_agent_returns_dict_for_lone_factory(config):
    from main import agent

    uid, data = factory_robot(uid="me", col=5, row=2, owner=0)
    obs = make_obs(robots={uid: data})

    actions = agent(obs, config)

    assert isinstance(actions, dict)
    assert set(actions.keys()) == {"me"}
    assert isinstance(actions["me"], str)


def test_agent_skips_enemy_units(config):
    from main import agent

    me_uid, me = factory_robot(uid="me", col=5, row=2, owner=0)
    enemy_uid, enemy = factory_robot(uid="enemy", col=14, row=2, owner=1)
    obs = make_obs(robots={me_uid: me, enemy_uid: enemy})

    actions = agent(obs, config)

    assert "enemy" not in actions
    assert "me" in actions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL — `main.agent` may exist but its current shape is undefined; we'll replace it.

- [ ] **Step 3: Replace `main.py` with the skeleton**

Overwrite `main.py`:

```python
"""Maze Crawler agent: heuristic FSM + BFS pathfinding."""

# === TUNABLES ===
SAFETY_MARGIN          = 4
SAFETY_HORIZON         = 3
LOW_ENERGY_RATIO       = 0.30
LATE_GAME_STOP_BUILD   = 30
LATE_GAME_HOARD        = 50
EXPLORE_KERNEL         = 5
NORTH_BIAS             = 1.5
WALL_DETOUR_THRESHOLD  = 6
ROLE_REASSIGN_PERIOD   = 8
TRANSFER_OVERFLOW_GAP  = 5
CRYSTAL_DETOUR_BUDGET  = 2
ENEMY_NEAR_RADIUS      = 2
MINER_REACH_LIMIT      = 25
BFS_MAX_DIST           = 80

# === CONSTANTS ===
TYPE_FACTORY = 0
TYPE_SCOUT = 1
TYPE_WORKER = 2
TYPE_MINER = 3

WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8
DIR_TO_BIT = {"NORTH": WALL_N, "EAST": WALL_E, "SOUTH": WALL_S, "WEST": WALL_W}
DIR_OFFSETS = {"NORTH": (0, 1), "EAST": (1, 0), "SOUTH": (0, -1), "WEST": (-1, 0)}
OPPOSITE_DIR = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}

# === MODULE-LEVEL MEMORY ===
_MEM: dict = {
    "walls": {},
    "mines": {},
    "mining_nodes": set(),
    "roles": {},
    "targets": {},
    "last_actions": {},
    "turn": 0,
    "enemy_factory_seen": None,
}


def agent(obs, config):
    """Entry point. Returns dict of {uid: action_str} for our units only."""
    actions = {}
    for uid, data in obs.robots.items():
        if data[4] != obs.player:
            continue
        actions[uid] = "IDLE"
    return actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/__init__.py tests/conftest.py tests/test_smoke.py
git commit -m "feat: scaffold agent skeleton with tunables and smoke tests"
```

---

## Task 2: Memory Layer — Coordinate Translation & Wall Merge

**Files:**
- Modify: `main.py` (add memory functions after `_MEM`)
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL — `memory_update` not defined.

- [ ] **Step 3: Implement memory_update**

Add to `main.py` after `_MEM` definition:

```python
def memory_update(obs, config, mem):
    """Single writer for persistent state. Idempotent per-turn merge."""
    width = config.width
    south = obs.southBound

    # 1) Prune anything that scrolled off
    mem["walls"]  = {k: v for k, v in mem["walls"].items()  if k[1] >= south}
    mem["mines"]  = {k: v for k, v in mem["mines"].items()  if k[1] >= south}
    mem["mining_nodes"] = {c for c in mem["mining_nodes"] if c[1] >= south}

    # 2) Merge wall info from this frame's obs
    for idx, val in enumerate(obs.walls):
        if val == -1:
            continue
        col = idx % width
        row = idx // width + south
        mem["walls"][(col, row)] = val

    # 3) Merge mine info
    for key, data in obs.mines.items():
        c, r = (int(x) for x in key.split(","))
        if r < south:
            continue
        mem["mines"][(c, r)] = tuple(data)

    # 4) Merge mining nodes (only currently visible per spec)
    for key in obs.miningNodes:
        c, r = (int(x) for x in key.split(","))
        if r >= south:
            mem["mining_nodes"].add((c, r))

    # 5) Track enemy factory sightings
    for uid, d in obs.robots.items():
        if d[0] == TYPE_FACTORY and d[4] != obs.player:
            mem["enemy_factory_seen"] = (d[1], d[2])

    # 6) Drop roles/targets for vanished UIDs
    live = set(obs.robots.keys())
    mem["roles"]   = {u: r for u, r in mem["roles"].items()   if u in live}
    mem["targets"] = {u: t for u, t in mem["targets"].items() if u in live}

    # 7) Tick
    mem["turn"] += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_memory.py
git commit -m "feat(memory): add wall/mine/role merge and pruning"
```

---

## Task 3: Context Builder

**Files:**
- Modify: `main.py` (add `Context` dataclass and `build_context`)
- Modify: `tests/test_memory.py` (append context tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory.py -v`
Expected: 3 new FAILs — `build_context` undefined.

- [ ] **Step 3: Implement Context and build_context**

Add to `main.py` after `memory_update`:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Unit:
    uid: str
    type: int
    col: int
    row: int
    energy: int
    owner: int
    move_cd: int
    jump_cd: int
    build_cd: int

    @property
    def cell(self):
        return (self.col, self.row)


@dataclass(frozen=True)
class Context:
    obs: Any
    config: Any
    mem: dict
    turn: int
    south: int
    north: int
    width: int
    me: int
    walls: dict
    crystals: dict
    mines: dict
    nodes: set
    my_factory: Any
    my_units: tuple
    enemy_units: tuple
    enemy_factory: Any
    danger_rows: frozenset


def _parse_unit(uid, data):
    return Unit(
        uid=uid,
        type=data[0], col=data[1], row=data[2], energy=data[3], owner=data[4],
        move_cd=data[5] if len(data) > 5 else 0,
        jump_cd=data[6] if len(data) > 6 else 0,
        build_cd=data[7] if len(data) > 7 else 0,
    )


def _parse_dict_keys(d):
    return {tuple(int(x) for x in k.split(",")): v for k, v in d.items()}


def build_context(obs, config, mem):
    me = obs.player
    units = [_parse_unit(uid, d) for uid, d in obs.robots.items()]
    my_units = tuple(u for u in units if u.owner == me)
    enemy_units = tuple(u for u in units if u.owner != me)
    my_factory = next((u for u in my_units if u.type == TYPE_FACTORY), None)
    enemy_factory = next((u for u in enemy_units if u.type == TYPE_FACTORY), None)

    danger_rows = frozenset(
        range(obs.southBound, obs.southBound + SAFETY_HORIZON)
    )

    return Context(
        obs=obs,
        config=config,
        mem=mem,
        turn=mem["turn"],
        south=obs.southBound,
        north=obs.northBound,
        width=config.width,
        me=me,
        walls=dict(mem["walls"]),
        crystals=_parse_dict_keys(obs.crystals),
        mines=dict(mem["mines"]),
        nodes=set(mem["mining_nodes"]),
        my_factory=my_factory,
        my_units=my_units,
        enemy_units=enemy_units,
        enemy_factory=enemy_factory,
        danger_rows=danger_rows,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory.py -v`
Expected: all PASS (5 + 3).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_memory.py
git commit -m "feat(context): add per-turn read-only context + Unit dataclass"
```

---

## Task 4: Passable Predicate + Fixed-Wall Detection

**Files:**
- Modify: `main.py` (add `is_fixed_wall`, `passable_strict`, `passable_loose`)
- Create: `tests/test_passable.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_passable.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_passable.py -v`
Expected: FAIL — `is_fixed_wall`, `passable_strict`, `passable_loose` undefined.

- [ ] **Step 3: Implement predicates**

Add to `main.py` after `build_context`:

```python
def is_fixed_wall(config, cell, direction):
    """Perimeter outer walls and central mirror axis are fixed (cannot be modified)."""
    col, _ = cell
    width = config.width
    half = width // 2
    if direction == "WEST" and col == 0:
        return True
    if direction == "EAST" and col == width - 1:
        return True
    if direction == "EAST" and col == half - 1:
        return True
    if direction == "WEST" and col == half:
        return True
    return False


def _wall_between(ctx, cell, direction):
    """True if a wall blocks moving in `direction` from `cell` (using known map)."""
    val = ctx.walls.get(cell)
    if val is None:
        return False  # treat unknown as passable (optimistic)
    return bool(val & DIR_TO_BIT[direction])


def _in_bounds(ctx, cell):
    c, r = cell
    return 0 <= c < ctx.width and ctx.south <= r <= ctx.north


def _step(cell, direction):
    dc, dr = DIR_OFFSETS[direction]
    return (cell[0] + dc, cell[1] + dr)


def passable_loose(ctx, cell, direction):
    """Wall + bounds check, ignores danger_rows."""
    if _wall_between(ctx, cell, direction):
        return False
    nxt = _step(cell, direction)
    return _in_bounds(ctx, nxt)


def passable_strict(ctx, cell, direction):
    """Wall + bounds + danger_rows check."""
    if not passable_loose(ctx, cell, direction):
        return False
    nxt = _step(cell, direction)
    return nxt[1] not in ctx.danger_rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_passable.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_passable.py
git commit -m "feat(context): add fixed-wall detection and passable predicates"
```

---

## Task 5: BFS

**Files:**
- Modify: `main.py` (add `bfs`)
- Create: `tests/test_bfs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bfs.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bfs.py -v`
Expected: FAIL — `bfs` undefined.

- [ ] **Step 3: Implement bfs**

Add to `main.py` after the passable predicates:

```python
from collections import deque


def bfs(start, goal_predicate, *, passable_fn,
        occupied=frozenset(), max_dist=BFS_MAX_DIST):
    """BFS over (col,row). Returns (distance, first_step_dir) or None."""
    if goal_predicate(start):
        return (0, "IDLE")

    # parent[cell] = (prev_cell, direction_taken_from_prev)
    parent = {start: (None, None)}
    queue = deque([(start, 0)])

    while queue:
        cell, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for direction in ("NORTH", "EAST", "SOUTH", "WEST"):
            if not passable_fn(cell, direction):
                continue
            nxt = _step(cell, direction)
            if nxt in parent or nxt in occupied:
                continue
            parent[nxt] = (cell, direction)
            if goal_predicate(nxt):
                # walk back to find first step from start
                step_dir = direction
                cursor = cell
                while parent[cursor][0] is not None:
                    step_dir = parent[cursor][1]
                    cursor = parent[cursor][0]
                return (dist + 1, step_dir)
            queue.append((nxt, dist + 1))

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bfs.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_bfs.py
git commit -m "feat(bfs): add BFS pathfinding with goal predicate and occupied set"
```

---

## Task 6: Tactics — Death Filter + Predict Next Cell

**Files:**
- Modify: `main.py` (add `predict_next_cell`, `death_filter`, `direction_score`)
- Create: `tests/test_tactics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tactics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tactics.py -v`
Expected: FAIL — functions undefined.

- [ ] **Step 3: Implement helpers**

Add to `main.py` after `bfs`:

```python
CRUSH_RANK = {TYPE_SCOUT: 1, TYPE_WORKER: 2, TYPE_MINER: 3, TYPE_FACTORY: 4}

# Movement-style action set (used to identify cells the unit will end on)
MOVEMENT_ACTIONS = {"NORTH", "EAST", "SOUTH", "WEST"}
JUMP_ACTIONS = {"JUMP_NORTH", "JUMP_EAST", "JUMP_SOUTH", "JUMP_WEST"}
BUILD_FACTORY_ACTIONS = {"BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"}

DIRECTION_SCORE = {"NORTH": 4, "EAST": 2, "WEST": 2, "SOUTH": 1, "IDLE": 0}


def direction_score(direction):
    return DIRECTION_SCORE.get(direction, 0)


def predict_next_cell(unit, action):
    """Cell the unit will occupy after this action resolves."""
    if action in MOVEMENT_ACTIONS:
        return _step(unit.cell, action)
    if action in JUMP_ACTIONS:
        d = action[len("JUMP_"):]
        dc, dr = DIR_OFFSETS[d]
        return (unit.col + 2 * dc, unit.row + 2 * dr)
    if action in BUILD_FACTORY_ACTIONS:
        return _step(unit.cell, "NORTH")
    return unit.cell


def _legal_movement_actions(ctx, unit, *, loose=False):
    """Return the subset of N/E/S/W movement actions that aren't wall-blocked or off-board."""
    fn = passable_loose if loose else passable_strict
    legal = []
    for d in ("NORTH", "EAST", "SOUTH", "WEST"):
        if fn(ctx, unit.cell, d):
            legal.append(d)
    return legal


def death_filter(ctx, unit, candidates, reservations, *,
                 reservation_types=None, enemy_cells=None):
    """Drop candidates that send `unit` to certain death."""
    if reservation_types is None:
        reservation_types = {}
    if enemy_cells is None:
        enemy_cells = {(e.col, e.row): e.type for e in ctx.enemy_units}

    survivors = []
    my_rank = CRUSH_RANK.get(unit.type, 0)

    for action in candidates:
        if action == "IDLE":
            survivors.append(action)
            continue
        nxt = predict_next_cell(unit, action)
        # off board check (movements only; specials stay put and were caught above)
        if action in MOVEMENT_ACTIONS:
            if not (0 <= nxt[0] < ctx.width and ctx.south <= nxt[1] <= ctx.north):
                continue
        # higher-crush enemy already there
        enemy_t = enemy_cells.get(nxt)
        if enemy_t is not None and CRUSH_RANK.get(enemy_t, 0) > my_rank:
            continue
        # reserved by stronger or equal-type friendly
        res_t = reservation_types.get(nxt)
        if res_t is not None:
            res_rank = CRUSH_RANK.get(res_t, 0)
            if res_rank > my_rank or (res_t == unit.type and unit.type != TYPE_FACTORY):
                continue
        survivors.append(action)
    return survivors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tactics.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_tactics.py
git commit -m "feat(tactics): add death filter, next-cell prediction, direction score"
```

---

## Task 7: Strategy — Frontier Scoring + Role Assignment

**Files:**
- Modify: `main.py` (add `frontier_score`, `assign_roles`)
- Modify: `tests/test_tactics.py` (append role tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tactics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tactics.py -v`
Expected: 3 new FAILs.

- [ ] **Step 3: Implement strategy helpers**

Add to `main.py` after `death_filter`:

```python
def frontier_score(ctx, cell):
    """E1 + E2: count -1 cells in EXPLORE_KERNEL window, weighted by north bias."""
    half = EXPLORE_KERNEL // 2
    unknown = 0
    for dc in range(-half, half + 1):
        for dr in range(-half, half + 1):
            probe = (cell[0] + dc, cell[1] + dr)
            if not (0 <= probe[0] < ctx.width):
                continue
            if not (ctx.south <= probe[1] <= ctx.north):
                continue
            if probe not in ctx.walls:
                unknown += 1
    span = max(ctx.north - ctx.south, 1)
    bias = 1.0 + NORTH_BIAS * (cell[1] - ctx.south) / span
    return unknown * bias


def _has_wall_bottleneck(ctx, factory):
    """Cheap heuristic: is there at least one known non-fixed wall in factory column above?"""
    if factory is None:
        return False
    for r in range(factory.row + 1, min(factory.row + WALL_DETOUR_THRESHOLD,
                                        ctx.north) + 1):
        val = ctx.walls.get((factory.col, r))
        if val is None:
            continue
        if val & WALL_N and not is_fixed_wall(ctx.config, (factory.col, r), "NORTH"):
            return True
    return False


def assign_roles(ctx):
    """Assign roles, respecting stickiness within ROLE_REASSIGN_PERIOD."""
    roles = dict(ctx.mem["roles"])
    reassess = (ctx.turn % ROLE_REASSIGN_PERIOD) == 0

    bottleneck = _has_wall_bottleneck(ctx, ctx.my_factory)
    have_node = bool(ctx.nodes)

    for u in ctx.my_units:
        if u.type == TYPE_FACTORY:
            roles[u.uid] = "FACTORY"
            continue
        if u.uid in roles and not reassess:
            continue
        if u.type == TYPE_SCOUT:
            roles[u.uid] = "EXPLORER"
        elif u.type == TYPE_WORKER:
            roles[u.uid] = "SAPPER" if bottleneck else "GUARD"
        elif u.type == TYPE_MINER:
            roles[u.uid] = "HARVESTER" if have_node else "GUARD"

    ctx.mem["roles"] = roles
    return roles
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tactics.py -v`
Expected: all PASS (6 + 3).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_tactics.py
git commit -m "feat(strategy): add frontier scoring and role assignment"
```

---

## Task 8: Strategy — Targets + Build Decision

**Files:**
- Modify: `main.py` (add `assign_targets`, `pick_factory_build`)
- Modify: `tests/test_tactics.py` (append target/build tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tactics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tactics.py -v`
Expected: 4 new FAILs.

- [ ] **Step 3: Implement target + build helpers**

Add to `main.py` after `assign_roles`:

```python
def pick_factory_build(ctx, factory):
    """B1..B4: choose a BUILD_* action or return None."""
    if factory is None or factory.build_cd > 0:
        return None
    cfg = ctx.config
    # B4 late-game stop
    if cfg.episodeSteps - ctx.turn < LATE_GAME_STOP_BUILD:
        return None
    # F3 build throttle
    if factory.energy < cfg.factoryEnergy * LOW_ENERGY_RATIO:
        return None

    have_scout = any(u.type == TYPE_SCOUT for u in ctx.my_units)
    have_miner = any(u.type == TYPE_MINER for u in ctx.my_units)
    bottleneck = _has_wall_bottleneck(ctx, factory)
    have_node = bool(ctx.nodes)

    # B1: first build always SCOUT
    if not have_scout and factory.energy >= cfg.scoutCost:
        return "BUILD_SCOUT"
    # B3: wall bottleneck → WORKER
    if bottleneck and factory.energy >= cfg.workerCost:
        have_worker = any(u.type == TYPE_WORKER for u in ctx.my_units)
        if not have_worker:
            return "BUILD_WORKER"
    # B2: reachable mining node → MINER
    if have_node and not have_miner and factory.energy >= cfg.minerCost:
        return "BUILD_MINER"
    # otherwise more SCOUTS
    if factory.energy >= cfg.scoutCost:
        return "BUILD_SCOUT"
    return None


def _passable_loose_fn(ctx):
    return lambda cell, d: passable_loose(ctx, cell, d)


def _passable_strict_fn(ctx):
    return lambda cell, d: passable_strict(ctx, cell, d)


def assign_targets(ctx):
    """Pick target cells per role and write into mem['targets']."""
    targets = {}
    roles = ctx.mem["roles"]
    pass_fn = _passable_loose_fn(ctx)

    for u in ctx.my_units:
        role = roles.get(u.uid)
        if role in (None, "FACTORY"):
            continue
        if role == "HARVESTER":
            best = None
            best_d = None
            for node in ctx.nodes:
                r = bfs(u.cell, lambda c, n=node: c == n,
                        passable_fn=pass_fn,
                        max_dist=MINER_REACH_LIMIT)
                if r is None:
                    continue
                d, _ = r
                if best_d is None or d < best_d:
                    best_d, best = d, node
            if best is not None:
                targets[u.uid] = best
            continue
        if role == "EXPLORER":
            # Pick a known cell adjacent to an unknown cell, scored by frontier_score.
            best = None
            best_score = -1
            for cell in ctx.walls:
                if any(
                    (cell[0] + dc, cell[1] + dr) not in ctx.walls
                    for dc, dr in DIR_OFFSETS.values()
                ):
                    s = frontier_score(ctx, cell)
                    if s > best_score:
                        best_score, best = s, cell
            if best is not None:
                targets[u.uid] = best
            continue
        if role == "GUARD":
            if ctx.my_factory is not None:
                targets[u.uid] = (ctx.my_factory.col, ctx.my_factory.row + 1)
            continue
        if role == "SAPPER":
            # Walk the column above the factory; first known wall on the path is target.
            if ctx.my_factory is not None:
                fc = ctx.my_factory.col
                for r in range(ctx.my_factory.row,
                               min(ctx.north, ctx.my_factory.row + WALL_DETOUR_THRESHOLD) + 1):
                    val = ctx.walls.get((fc, r))
                    if val is None:
                        continue
                    if val & WALL_N and not is_fixed_wall(ctx.config, (fc, r), "NORTH"):
                        targets[u.uid] = (fc, r)
                        break
            continue

    ctx.mem["targets"] = targets
    return targets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tactics.py -v`
Expected: all PASS (9 + 4).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_tactics.py
git commit -m "feat(strategy): add build decision and target assignment"
```

---

## Task 9: Per-Unit Decision Pipeline

**Files:**
- Modify: `main.py` (add `decide_unit`, `predict_reservation_type`)
- Modify: `tests/test_tactics.py` (append decision tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tactics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tactics.py -v`
Expected: 4 new FAILs.

- [ ] **Step 3: Implement decide_unit**

Add to `main.py` after `assign_targets`:

```python
def _legal_factory_intents(ctx, factory):
    """Build candidate actions for the factory in priority order."""
    candidates = []
    # F1 emergency: factory near south boundary → push NORTH first
    near_south = factory.row - ctx.south < SAFETY_MARGIN
    # F2 JUMP if NORTH is walled and worker can't break it in 2 turns
    blocked_north = _wall_between(ctx, factory.cell, "NORTH")
    if near_south and not blocked_north:
        candidates.append("NORTH")
    if blocked_north and factory.jump_cd == 0:
        # JUMP only if landing in bounds and not in danger_rows for strict
        landing = (factory.col, factory.row + 2)
        if (
            ctx.south <= landing[1] <= ctx.north
            and landing[1] not in ctx.danger_rows
        ):
            candidates.append("JUMP_NORTH")
    build = pick_factory_build(ctx, factory)
    if build is not None:
        candidates.append(build)
    if not blocked_north:
        candidates.append("NORTH")
    candidates.append("IDLE")
    # de-dup while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _bfs_first_step(ctx, unit, target, *, loose=False):
    fn = _passable_loose_fn(ctx) if loose else _passable_strict_fn(ctx)
    res = bfs(unit.cell, lambda c: c == target, passable_fn=fn)
    if res is None:
        return None
    return res[1]


def decide_unit(ctx, unit, reservations, reservation_types):
    """Per-unit decision pipeline. Always returns a legal action string."""
    role = ctx.mem["roles"].get(unit.uid, "GUARD")
    target = ctx.mem["targets"].get(unit.uid)

    # Factory has its own intent ladder
    if unit.type == TYPE_FACTORY:
        candidates = _legal_factory_intents(ctx, unit)
        survivors = death_filter(
            ctx, unit, candidates, reservations,
            reservation_types=reservation_types,
        )
        if survivors:
            return survivors[0]
        return "IDLE"

    # HARVESTER: TRANSFORM if on node
    if role == "HARVESTER" and unit.cell in ctx.nodes \
            and unit.energy >= ctx.config.transformCost:
        return "TRANSFORM"

    # SAPPER: BUILD/REMOVE if at the target wall and not fixed
    if role == "SAPPER" and target is not None and unit.cell == target \
            and unit.energy >= ctx.config.wallRemoveCost:
        # Determine which side has the bottleneck wall (default NORTH)
        for d in ("NORTH", "EAST", "SOUTH", "WEST"):
            if _wall_between(ctx, unit.cell, d) and not is_fixed_wall(
                ctx.config, unit.cell, d
            ):
                return f"REMOVE_{d}"
        # Fall through to movement if no removable wall here.

    # Default: BFS toward target, fall back to direction score.
    direction = None
    if target is not None:
        direction = _bfs_first_step(ctx, unit, target,
                                    loose=(role == "EXPLORER"))

    legal = _legal_movement_actions(
        ctx, unit, loose=(role == "EXPLORER")
    )
    candidates = []
    if direction in legal:
        candidates.append(direction)
    # backup directions ordered by direction_score
    for d in sorted(legal, key=direction_score, reverse=True):
        if d not in candidates:
            candidates.append(d)
    candidates.append("IDLE")

    survivors = death_filter(ctx, unit, candidates, reservations,
                             reservation_types=reservation_types)
    return survivors[0] if survivors else "IDLE"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tactics.py -v`
Expected: all PASS (13 + 4).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_tactics.py
git commit -m "feat(tactics): add per-unit decision pipeline with role intents"
```

---

## Task 10: Wiring — `agent` Conductor + Reservation Loop

**Files:**
- Modify: `main.py` (rewrite `agent` to call all layers in order)
- Modify: `tests/test_smoke.py` (append integration assertions)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_smoke.py`:

```python
def test_agent_full_pipeline_two_units(config):
    from main import agent

    me_fac_uid, me_fac = factory_robot(uid="f", col=5, row=2,
                                       energy=1000, owner=0)
    scout_uid, scout = "s", [1, 5, 3, 100, 0, 0, 0, 0]
    obs = make_obs(robots={me_fac_uid: me_fac, scout_uid: scout})

    actions = agent(obs, config)

    assert set(actions.keys()) == {"f", "s"}
    valid_prefixes = (
        "NORTH", "SOUTH", "EAST", "WEST", "IDLE",
        "BUILD_", "JUMP_", "REMOVE_", "TRANSFER_", "TRANSFORM",
    )
    for v in actions.values():
        assert any(v.startswith(p) for p in valid_prefixes), v


def test_agent_does_not_walk_off_board(config):
    from main import agent

    # Factory at the very north edge — must not pick NORTH.
    me_fac_uid, me_fac = factory_robot(uid="f", col=5, row=19,
                                       energy=1000, owner=0)
    obs = make_obs(robots={me_fac_uid: me_fac}, north=19)
    actions = agent(obs, config)
    assert actions["f"] != "NORTH"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_smoke.py -v`
Expected: 2 new FAILs (agent currently returns IDLE for everything).

- [ ] **Step 3: Replace `agent` with the full pipeline**

Replace the `agent` function in `main.py`:

```python
def agent(obs, config):
    """Entry point. Drives memory → context → strategy → tactics."""
    memory_update(obs, config, _MEM)
    ctx = build_context(obs, config, _MEM)

    assign_roles(ctx)
    assign_targets(ctx)

    actions = {}
    reservations = set()
    reservation_types = {}

    type_priority = {TYPE_FACTORY: 0, TYPE_MINER: 1, TYPE_WORKER: 2,
                     TYPE_SCOUT: 3}

    # Pre-reserve factory escort cells
    if ctx.my_factory is not None:
        f = ctx.my_factory
        reservations.add((f.col, f.row + 1))

    for unit in sorted(ctx.my_units,
                       key=lambda u: type_priority.get(u.type, 9)):
        action = decide_unit(ctx, unit, reservations, reservation_types)
        actions[unit.uid] = action
        nxt = predict_next_cell(unit, action)
        reservations.add(nxt)
        reservation_types[nxt] = unit.type

    # Record intent for next-turn coherence
    _MEM["last_actions"] = dict(actions)
    return actions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_smoke.py
git commit -m "feat(agent): wire memory → context → strategy → tactics pipeline"
```

---

## Task 11: Energy Economy — TRANSFER Overflow + Crystal Detour

**Files:**
- Modify: `main.py` (add `maybe_transfer`, integrate crystal detour into `assign_targets`)
- Modify: `tests/test_tactics.py` (append energy tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tactics.py`:

```python
def test_transfer_when_adjacent_to_factory_overflow(config):
    from main import maybe_transfer

    me_fac = factory_robot(uid="f", col=5, row=2, energy=1000, owner=0)
    scout = ("s", [1, 5, 3, 99, 0, 0, 0, 0])  # max_energy 100, gap = 1
    ctx = _ctx_with_units(config, my=[me_fac, scout])
    scout_unit = next(u for u in ctx.my_units if u.uid == "s")
    action = maybe_transfer(ctx, scout_unit)
    assert action == "TRANSFER_SOUTH"  # scout at (5,3) → factory at (5,2)


def test_transfer_skipped_when_low(config):
    from main import maybe_transfer

    me_fac = factory_robot(uid="f", col=5, row=2, energy=1000, owner=0)
    scout = ("s", [1, 5, 3, 50, 0, 0, 0, 0])
    ctx = _ctx_with_units(config, my=[me_fac, scout])
    scout_unit = next(u for u in ctx.my_units if u.uid == "s")
    assert maybe_transfer(ctx, scout_unit) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tactics.py -v`
Expected: 2 new FAILs.

- [ ] **Step 3: Implement maybe_transfer and integrate**

Add to `main.py` after `decide_unit`:

```python
_UNIT_MAX_ENERGY = {
    TYPE_SCOUT: "scoutMaxEnergy",
    TYPE_WORKER: "workerMaxEnergy",
    TYPE_MINER: "minerMaxEnergy",
}


def _max_energy_for(ctx, unit):
    attr = _UNIT_MAX_ENERGY.get(unit.type)
    if attr is None:
        return None  # factory has no cap
    return getattr(ctx.config, attr)


def maybe_transfer(ctx, unit):
    """M1: if adjacent to factory and overflowing, TRANSFER toward it."""
    if unit.type == TYPE_FACTORY or ctx.my_factory is None:
        return None
    fac = ctx.my_factory
    dx = fac.col - unit.col
    dy = fac.row - unit.row
    if abs(dx) + abs(dy) != 1:
        return None
    direction = None
    if dx == 1:
        direction = "EAST"
    elif dx == -1:
        direction = "WEST"
    elif dy == 1:
        direction = "NORTH"
    elif dy == -1:
        direction = "SOUTH"
    if direction is None or _wall_between(ctx, unit.cell, direction):
        return None

    cap = _max_energy_for(ctx, unit)
    overflow_trigger = (
        cap is not None and unit.energy >= cap - TRANSFER_OVERFLOW_GAP
    )
    factory_low = fac.energy < ctx.config.factoryEnergy * LOW_ENERGY_RATIO
    if overflow_trigger or factory_low:
        return f"TRANSFER_{direction}"
    return None
```

Wire it into `decide_unit` at the very top, just after the role lookup:

```python
    # M1: TRANSFER overflow / factory help — overrides movement when it fires.
    if unit.type != TYPE_FACTORY:
        t = maybe_transfer(ctx, unit)
        if t is not None:
            return t
```

(Insert immediately after `target = ctx.mem["targets"].get(unit.uid)` in `decide_unit`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_tactics.py
git commit -m "feat(economy): add TRANSFER overflow back to factory"
```

---

## Task 12: Local Match Harness — vs Random

**Files:**
- Create: `tests/match_random.py`

- [ ] **Step 1: Add the script**

Create `tests/match_random.py`:

```python
"""Manual local match harness. Run: python tests/match_random.py"""

import sys

from kaggle_environments import make


SEEDS = [42, 7, 13, 99, 256]
WIN_THRESHOLD = 0.8


def play_one(seed):
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    env.run(["main.py", "random"])
    rewards = [env.steps[-1][i].reward for i in range(2)]
    if rewards[0] is None or rewards[1] is None:
        return None
    return rewards[0] > rewards[1]


def main():
    results = []
    for seed in SEEDS:
        outcome = play_one(seed)
        results.append(outcome)
        print(f"seed={seed} → {'WIN' if outcome else 'LOSS/TIE'}")
    wins = sum(1 for r in results if r)
    rate = wins / len(SEEDS)
    print(f"win rate: {rate:.0%} ({wins}/{len(SEEDS)})")
    sys.exit(0 if rate >= WIN_THRESHOLD else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the harness**

Run: `python tests/match_random.py`
Expected: prints per-seed outcomes; exit 0 if win rate ≥ 80%.

If win rate is below threshold, do not silently retune — report the failure to the reviewer with which seeds lost and a one-line hypothesis (e.g., "lost seed 7 — factory boxed in, JUMP not triggered"). Tuning happens in a follow-up iteration.

- [ ] **Step 3: Commit**

```bash
git add tests/match_random.py
git commit -m "test: add manual vs-random match harness"
```

---

## Task 13: Smoke Run from README + Submission Sanity

**Files:** none (verification only)

- [ ] **Step 1: Run the README quick-start**

Run:

```bash
python -c "from kaggle_environments import make; env=make('crawl', configuration={'randomSeed': 42}, debug=True); env.run(['main.py', 'random']); print([(i, s.reward) for i, s in enumerate(env.steps[-1])])"
```

Expected: prints two `(idx, reward)` tuples without exception. Player 0 (us) reward should be a number, not None.

- [ ] **Step 2: Verify single-file submission shape**

Run: `python -c "import ast; ast.parse(open('main.py').read())"`
Expected: no output (parse OK).

Run: `python -c "from main import agent; from tests.conftest import make_obs, make_config, factory_robot; uid,d=factory_robot(); print(agent(make_obs(robots={uid:d}), make_config()))"`
Expected: prints a `{uid: action_str}` dict.

- [ ] **Step 3: Commit (no-op verification)**

No commit needed for verification. Move on.

---

## Self-Review

**Spec coverage:**

| Spec section | Implemented in |
|--------------|---------------|
| §3 Memory layer (walls / mines / nodes / roles / targets / pruning) | Task 2 |
| §4 Context + passable + BFS | Tasks 3-5 |
| §5.1 F1/F2/F3 factory survival | Task 9 (`_legal_factory_intents`) + Task 8 (F3 in `pick_factory_build`) |
| §5.2 B1/B2/B3/B4 build decisions | Task 8 (`pick_factory_build`) |
| §5.3 R1/R2/R3/R4 role assignment + stickiness | Task 7 (`assign_roles`) |
| §5.4 E1/E2 frontier scoring + north bias | Task 7 (`frontier_score`) |
| §5.4 E3 mid-line door probe | not implemented in v1 — flagged as open risk in spec §10; deferred. |
| §5.5 M1 TRANSFER overflow | Task 11 (`maybe_transfer`) |
| §5.5 M2 crystal detour | not implemented in v1 — defer; spec lists as "recommended" not "required". |
| §5.5 M3 abandon mine threshold | not implemented in v1 — defer. |
| §5.6 C1 friendly-collision prevention | Task 6 + Task 10 (reservation loop + `death_filter`) |
| §5.6 C2-C4 enemy-near caution / anti-scout / factory self-defense | not in v1 — defer; "optional" in spec. |
| §5.7 L1-L3 late game | partial: B4 stop-build (Task 8). L1/L2/L3 deferred. |
| §6.1 scheduling order | Task 10 |
| §6.2 per-unit pipeline (death filter → role → BFS → degrade) | Task 9 |
| §6.3 degradation rules | Task 9 (BFS None / fixed wall guard / Miner node missing implicit via target staleness) |
| §6.4 factory escort | Task 10 (pre-reserve `(f.col, f.row+1)`) |
| §7 Tunables | Task 1 |
| §8 Testing strategy: smoke / BFS / match | Tasks 1, 5, 12 |

**Deferred items** are listed under "Open Risks" in the spec and are explicitly low-priority. They can be picked up in a follow-up plan once v1 has a baseline win rate.

**Placeholder scan:** No TBD/TODO/"add appropriate handling" remain. All code blocks contain real code; all step counts are explicit; all expected outputs are spelled out.

**Type consistency:** `Unit`, `Context`, `_MEM`, `pick_factory_build`, `assign_roles`, `assign_targets`, `decide_unit`, `predict_next_cell`, `death_filter`, `bfs`, `frontier_score`, `maybe_transfer`, `is_fixed_wall`, `passable_strict`, `passable_loose` — all referenced names match their definitions.

**Decoded edge cases:**
- `_wall_between` returns `False` for unknown cells (optimistic) so BFS doesn't refuse to explore unmapped frontier; confirmed safe because `passable_loose` still requires `_in_bounds`.
- `death_filter` treats same-type friendly reservation as deadly (mutual destroy), per spec §6.2.
- `pick_factory_build` checks `factory.build_cd` before any other gate, matching engine rules.
