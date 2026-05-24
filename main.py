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


def memory_update(obs, config, mem):
    """Single writer for persistent state. Idempotent per-turn merge."""
    width = config.width
    south = obs.southBound

    # 1) Prune anything that scrolled off
    mem["walls"] = {k: v for k, v in mem["walls"].items() if k[1] >= south}
    mem["mines"] = {k: v for k, v in mem["mines"].items() if k[1] >= south}
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
    mem["roles"] = {u: r for u, r in mem["roles"].items() if u in live}
    mem["targets"] = {u: t for u, t in mem["targets"].items() if u in live}

    # 7) Tick
    mem["turn"] += 1


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
    """Wall + bounds + danger_rows check.

    If the unit is already standing in a danger row, any safe move is allowed
    (don't lock the unit in place); otherwise reject moves into a danger row.
    """
    if not passable_loose(ctx, cell, direction):
        return False
    if cell[1] in ctx.danger_rows:
        return True
    nxt = _step(cell, direction)
    return nxt[1] not in ctx.danger_rows


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


def agent(obs, config):
    """Entry point. Returns dict of {uid: action_str} for our units only."""
    actions = {}
    for uid, data in obs.robots.items():
        if data[4] != obs.player:
            continue
        actions[uid] = "IDLE"
    return actions
