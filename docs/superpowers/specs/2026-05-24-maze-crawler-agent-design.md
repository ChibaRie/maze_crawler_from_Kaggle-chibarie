# Maze Crawler Agent — Design Spec

**Date:** 2026-05-24
**Target:** `main.py` — single-file submission for Kaggle `maze-crawler` competition
**Approach:** Heuristic FSM + BFS pathfinding + light risk evaluation
**Compute budget:** Light per turn; no rollouts, no multi-step minimax beyond local factory safety

## 1. Goals & Non-Goals

### Goals

- Produce a competitive single-file `agent(obs, config)` that beats the baseline `random` opponent reliably (target > 80% over 5 seeds).
- Survive to step 500 in the majority of games by keeping the factory clear of the southern boundary.
- Reach a positive expected score on the Kaggle skill ladder via robust factory survival, opportunistic energy economy, and basic enemy-aware defense.

### Non-Goals

- No reinforcement learning, no learned weights, no multi-file packaging.
- No reimplementation of the full game engine for rollout-based search.
- No deep multi-step minimax. Local 1-step risk evaluation only, and only for factory JUMP/move safety.

## 2. Architecture

Four layers in a strict downward dataflow. The top-level `agent` is the conductor.

```
┌─────────────────────────────────────────────────────────┐
│ agent(obs, config)                  ← conductor (entry) │
│   1. memory.update(obs)             ← persistence       │
│   2. ctx = build_context(obs, mem)  ← derived read-only │
│   3. plan = strategy.plan(ctx)      ← role assignment   │
│   4. actions = tactics.act(plan)    ← per-unit actions  │
│   5. memory.commit(actions)         ← record intent     │
│   return actions                                        │
└─────────────────────────────────────────────────────────┘
```

Lower layers do not write upper-layer state. The only writeback to memory is `commit`, which records actions issued this turn for next-turn coherence (e.g., role stickiness, deadlock detection).

A single module-level dict `_MEM` holds all persistent state. `memory.update` is the sole writer of fields other than `commit`.

## 3. Memory Layer

### Fields

| Field | Type | Purpose | Write / clear |
|------|------|---------|------------|
| `walls` | `dict[(col,row)] = bitfield` | Permanent known walls | merge from `obs.walls` where value != -1; rows below `southBound` are pruned |
| `mines` | `dict[(col,row)] = (energy, max, owner)` | Discovered mines (any owner) | merge from `obs.mines`; pruned with row |
| `mining_nodes` | `set[(col,row)]` | Historically seen node positions | added on sight; removed on TRANSFORM, on overwrite, or with row |
| `roles` | `dict[uid] = role` | Unit role (FACTORY, EXPLORER, HARVESTER, SAPPER, GUARD) | written by strategy; cleaned to only living UIDs |
| `targets` | `dict[uid] = (col,row)` | Current goal cell | written by strategy; cleared when reached or stale |
| `last_actions` | `dict[uid] = action_str` | Action issued last turn | written in `commit` |
| `turn` | `int` | Internal turn counter | incremented in `update` |
| `enemy_factory_seen` | `(col,row) or None` | Last known enemy factory cell | overwritten on sight |

### Coordinate convention

- All cells stored in `_MEM` use **global** `(col, row)` coordinates, where `row` is the absolute maze row (not relative to `southBound`).
- `obs.walls[index]` translates: `row = index // width + southBound; col = index % width`.
- This keeps memory consistent across scrolls.

### Pruning rule

At the top of `memory.update`:

```
for d in (walls, mines, mining_nodes):
    drop keys whose row < obs.southBound
roles  = {uid: r for uid, r in roles.items() if uid in obs.robots}
targets = {uid: t for uid, t in targets.items() if uid in obs.robots}
```

This bounds dictionary growth and prevents stale role assignment for dead units.

## 4. Context & BFS

### Context (rebuilt every turn, read-only)

A frozen dataclass exposing:

- raw `obs`, `config`, reference to `mem`
- `turn`, `south`, `north`, `width`, `me` (player index)
- `walls`: merged map from permanent memory + current-frame `obs.walls`
- `crystals`, `mines`, `nodes`: merged where appropriate
- `my_factory`, `my_units`, `enemy_units`, `enemy_factory`
- `danger_rows: set[int]` — the set of rows that will be scrolled away within `SAFETY_HORIZON` turns (computed from current `southBound` plus a forecast of the scroll-interval ramp)

### `passable` predicate

Two variants:

- `passable_strict(cell, dir)` — wall check + within `[south, north]` + within `[0, width)` + target row not in `danger_rows`. Used by factory and high-value units.
- `passable_loose(cell, dir)` — same minus the `danger_rows` check. Used by scouts that prefer information over short-term safety.

### BFS

```python
def bfs(start, goal_predicate, *, passable_fn,
        occupied=frozenset(), max_dist=BFS_MAX_DIST) -> tuple[int, str] | None
```

- Standard deque BFS over `(col, row)`.
- `goal_predicate(cell) -> bool` lets callers query "nearest mining node", "nearest unknown cell", etc.
- `occupied` excludes specific cells (not the start). Used to keep units out of reservations.
- Returns `(distance, first_step_dir)` where `first_step_dir ∈ {"NORTH","SOUTH","EAST","WEST"}`. `"IDLE"` if start already satisfies predicate.
- Returns `None` if no path within `max_dist`.

Per-turn cost is bounded: at most one BFS per unit, each O(width × visible_rows) ≈ 600 nodes, well under the 1 s/step Kaggle limit.

## 5. Strategy Layer — Sub-strategies

The strategy layer applies seven groups of sub-strategies in priority order. Section numbers below match the user-confirmed strategy list.

### 5.1 Factory survival (required)

- **F1 Safety margin**: factory must stay above `south + SAFETY_MARGIN`. If the factory is at or below this threshold, every other goal yields to the factory's northbound move.
- **F2 JUMP gate**: if a wall blocks NORTH, no SAPPER can break it within 2 turns, and `jump_cd == 0`, JUMP_NORTH is considered. Reject the JUMP if the landing cell is off-board, contains a hostile factory in a 1-step combat path, or lands in `danger_rows` with no follow-up move.
- **F3 Build throttle**: if `factory_energy < factoryEnergy * LOW_ENERGY_RATIO`, suspend BUILD_* actions and prioritize NORTH-ward motion.

### 5.2 Build decisions (required)

- **B1 Build priority sequence**: first build is always BUILD_SCOUT (cheapest, fastest vision). Subsequent builds chosen by current need: SAPPER if a wall is bottlenecking the route; HARVESTER if a reachable mining node exists and no miner is in flight; otherwise SCOUT to expand vision.
- **B2 Mining-node trigger**: a Miner is queued only if a `mining_nodes` entry has a BFS path with `dist < MINER_REACH_LIMIT` and no in-flight miner is targeting it.
- **B3 Wall-bottleneck trigger**: when BFS detour cost from any explorer to its target exceeds `WALL_DETOUR_THRESHOLD` due to a non-fixed wall, queue a Worker.
- **B4 Late-game stop**: if `episodeSteps - turn < LATE_GAME_STOP_BUILD`, suspend all builds.

### 5.3 Role assignment (required)

- **R1** Scout → EXPLORER.
- **R2** Worker → SAPPER if a target wall is en-route to a known goal; else GUARD (escort the factory).
- **R3** Miner → HARVESTER (locked to the nearest reachable node).
- **R4** Roles are reassessed every `ROLE_REASSIGN_PERIOD` turns or when a unit's target becomes unreachable / consumed. Roles are otherwise sticky to avoid oscillation.

### 5.4 Exploration (recommended)

- **E1 Unknown-density scoring**: for each candidate frontier cell (a known cell adjacent to `obs.walls == -1`), score = count of `-1` cells in an `EXPLORE_KERNEL × EXPLORE_KERNEL` window centered on it.
- **E2 North bias**: multiply E1 score by `1 + NORTH_BIAS * (row - south) / (north - south + 1)`.
- **E3 Mid-line door probe**: when the central mirror axis appears as wall in memory but at least one row has not been probed, send the highest-energy SCOUT to step adjacent to the axis to test for a door (E or W move attempt; the action is legal regardless and informs the wall map).

### 5.5 Energy economy (recommended)

- **M1 TRANSFER overflow**: a unit adjacent to the factory with `energy >= max_energy - TRANSFER_OVERFLOW_GAP`, or any unit adjacent to factory when factory has < `LOW_ENERGY_RATIO * factoryEnergy`, issues `TRANSFER_*` toward the factory.
- **M2 Crystal detour**: while routing, if a visible crystal is within `CRYSTAL_DETOUR_BUDGET` extra steps of the BFS path, retarget through it.
- **M3 Abandon-mine threshold**: a `mining_nodes` entry within `ENEMY_NEAR_RADIUS` of a known enemy factory or two enemy units is removed from candidate set for this turn.

### 5.6 Combat & defense (optional)

- **C1 Friendly-collision prevention**: tactics layer reserves predicted next-cells in priority order. A lower-priority unit whose only safe move would collide with a higher-priority friendly unit IDLEs instead.
- **C2 Enemy-near caution**: with an enemy unit within `ENEMY_NEAR_RADIUS` (Manhattan), Scouts retreat northwest/northeast away; Workers stop attempting BUILD_* actions adjacent to that enemy.
- **C3 Anti-Scout trade**: if a friendly Scout is adjacent to an enemy Scout and no friendly Scout of better positioning exists, allow the trade (mutual destroy is even).
- **C4 Factory self-defense**: if the enemy factory's `jump_cd <= 1` and lies within JUMP range of our factory, our factory pre-emptively side-steps if a safe lateral move exists.

### 5.7 Late game (optional)

- **L1 Energy hoarding**: in the last `LATE_GAME_HOARD` turns, all units route home and TRANSFER to factory each step they are adjacent.
- **L2 Unit-count preservation**: late game, do not send units on suicide objectives. A live IDLE unit is worth a tiebreaker count tick.
- **L3 Emergency JUMP**: if factory is one row from `south` and `NORTH` is blocked, JUMP_NORTH overrides cooldown checks (within game rules — i.e., only if `jump_cd == 0`; otherwise NORTH if possible).

## 6. Tactics Layer

### 6.1 Scheduling order

Per turn, units are processed in the order `(factory, miners, workers, scouts)`. Each unit's predicted next cell is added to `reservations` after it decides, and subsequent units must avoid that cell unless they can win the resulting crush.

```python
reservations: set[tuple[int,int]] = set()
for unit in sorted(my_units, key=type_priority):
    action = decide(unit, ctx, plan, reservations)
    actions[unit.uid] = action
    reservations.add(predict_next_cell(unit, action))
```

`predict_next_cell` returns the spawn cell for `BUILD_*`, the landing cell for `JUMP_*`, the target adjacent cell for movement, and the unit's own cell for `IDLE`, `TRANSFORM`, `BUILD_DIR`, `REMOVE_DIR`, `TRANSFER_*`.

### 6.2 Per-unit pipeline

```
1) Death filter: enumerate legal actions, drop those that
   - step off N/S edge,
   - step into a higher-crush enemy,
   - step into a higher-crush friendly already in reservations,
   - step into a same-type friendly already in reservations (mutual destroy).
2) Role-specific intent:
   - FACTORY: F1 → F2 → B1..B4 → NORTH or IDLE
   - HARVESTER: TRANSFORM if on node; else BFS to assigned node
   - SAPPER: BUILD_DIR/REMOVE_DIR if at the target wall; else BFS
   - EXPLORER: BFS to assigned frontier cell
   - GUARD: BFS to factory escort cell
3) Convert intent to a first-step direction (BFS) or special action.
4) If the intent's first step is filtered out by step 1:
   - re-rank the surviving moves by direction score (NORTH > E/W > SOUTH),
   - pick highest; if none, IDLE.
```

### 6.3 Degradation rules

The pipeline guarantees a legal action even if planning fails:

- BFS returns None → fall back to direction-score move; all blocked → IDLE.
- JUMP landing all unsafe → skip JUMP, retry next turn.
- Worker BUILD/REMOVE target is a fixed wall (perimeter or mirror axis) → skip the order; revert to GUARD.
- Miner reaches its node and node is gone → revert to GUARD.

### 6.4 Factory escort

When the factory is scheduled to move NORTH, both the factory's target cell and the cell two rows north are added to `reservations` at the start of the loop, so other units never block the factory's path.

## 7. Tunable Constants

A `# === TUNABLES ===` block at the top of `main.py`:

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

These are tuned via local self-play and not via remote services.

## 8. Testing Strategy

Three lightweight checks live in `tests/` (not packaged into the submission):

1. **Smoke test (`tests/test_agent.py`)**: build a synthetic `obs` namespace with a single factory, a few walls, and one crystal. Assert `agent(obs, config)` returns a dict whose keys are the friendly UIDs and whose values are members of the legal action vocabulary.
2. **BFS unit test**: assemble a small wall grid by hand. Assert correct first-step direction when a path exists, and `None` when fully walled off.
3. **Local match harness**: a script that runs `kaggle_environments.make("crawl")` against the `random` opponent on five seeds; passes if win rate ≥ 80%.

Coverage is not enforced for this single-file submission style; the repository's 80% rule applies to projects with a `src/` layout.

## 9. File Layout

```
main.py                  ← submission, single file
docs/superpowers/specs/  ← this spec and future ones
tests/test_agent.py      ← smoke + bfs unit tests, not packaged
```

The submission command remains:

```
kaggle competitions submit maze-crawler -f main.py -m "<message>"
```

## 10. Open Risks

- **Mirror axis door detection**: `doorProbability=0.08` per row gives sparse doors. E3 may yield slow returns; if local benches show < 5% improvement vs disabling E3, drop it.
- **JUMP safety evaluation**: a 1-step look-ahead misses two-turn enemy convergence on the landing cell. Acceptable risk given the light-budget scope.
- **Stuck Sappers**: if BUILD/REMOVE is misordered against a fixed wall, energy bleeds at 100/turn. The `is_fixed_wall` check at the perimeter and mirror axis is critical and must be unit-tested.
