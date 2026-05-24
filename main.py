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


def agent(obs, config):
    """Entry point. Returns dict of {uid: action_str} for our units only."""
    actions = {}
    for uid, data in obs.robots.items():
        if data[4] != obs.player:
            continue
        actions[uid] = "IDLE"
    return actions
