# Maze Crawler Agent — 设计文档

**日期：** 2026-05-24
**目标：** `main.py` —— Kaggle `maze-crawler` 竞赛单文件提交
**方案：** 启发式状态机 + BFS 寻路 + 轻量风险评估
**单回合预算：** 轻量；不做 rollout，不做超出"工厂安全评估"范围的多步 minimax

## 1. 目标与非目标

### 目标

- 产出一个有竞争力的 `agent(obs, config)` 单文件实现，对基线 `random` 对手稳定胜出（5 个种子目标胜率 > 80%）。
- 大多数对局能让工厂活到第 500 回合，关键是不被南界扫掉。
- 通过稳健的工厂保命、机会主义的能量经济、基础的敌情应对，在 Kaggle 天梯上获得正期望分。

### 非目标

- 不使用强化学习、不加载学习权重、不做多文件打包。
- 不重写完整游戏引擎做 rollout 搜索。
- 不做深层多步 minimax，只在工厂 JUMP/移动的局部安全评估上做 1 步前瞻。

## 2. 架构

四层结构，严格自顶向下数据流。顶层 `agent` 做总指挥。

```
┌─────────────────────────────────────────────────────────┐
│ agent(obs, config)                  ← 指挥层（顶层入口） │
│   1. memory.update(obs)             ← 持久化层           │
│   2. ctx = build_context(obs, mem)  ← 派生只读视图       │
│   3. plan = strategy.plan(ctx)      ← 战略层（角色分配） │
│   4. actions = tactics.act(plan)    ← 战术层（逐单位）   │
│   5. memory.commit(actions)         ← 记下我方意图       │
│   return actions                                        │
└─────────────────────────────────────────────────────────┘
```

下层不写上层状态。`commit` 是回写 memory 的唯一处，记录本回合下发动作以保证跨回合一致性（角色粘性、死锁检测）。

模块级单一字典 `_MEM` 持有全部持久状态。除 `commit` 外，`memory.update` 是 `_MEM` 唯一写入点。

## 3. 记忆层（Memory）

### 字段

| 字段 | 类型 | 用途 | 写入 / 清理时机 |
|------|------|------|----------------|
| `walls` | `dict[(col,row)] = bitfield` | 永久已知墙 | 当 `obs.walls` 该格 ≠ -1 时合并；行 < `southBound` 时剔除 |
| `mines` | `dict[(col,row)] = (energy, max, owner)` | 已发现矿（含敌方） | 从 `obs.mines` 合并；随行被扫剔除 |
| `mining_nodes` | `set[(col,row)]` | 历史见过的 node 位置 | 进入视野时记入；TRANSFORM 后或被覆盖时移除；随行被扫剔除 |
| `roles` | `dict[uid] = role` | 单位角色（FACTORY/EXPLORER/HARVESTER/SAPPER/GUARD） | 战略层写；只保留仍存活 UID |
| `targets` | `dict[uid] = (col,row)` | 单位当前目标格 | 战略层写；到达或失效时清 |
| `last_actions` | `dict[uid] = action_str` | 上回合下发的动作 | `commit` 时写 |
| `turn` | `int` | 内部回合计数 | `update` 中递增 |
| `enemy_factory_seen` | `(col,row) or None` | 见过敌方工厂的最后位置 | 看到时覆盖 |

### 坐标约定

- `_MEM` 中所有格子使用**全局** `(col, row)` 坐标，`row` 是绝对行号（不是相对 `southBound` 的偏移）。
- `obs.walls[index]` 翻译规则：`row = index // width + southBound; col = index % width`。
- 这样能跨"南界滚动"保持记忆一致。

### 剪枝规则

`memory.update` 开头：

```
对 walls / mines / mining_nodes：删除所有 row < obs.southBound 的键
roles   = {uid: r for uid, r in roles.items()   if uid in obs.robots}
targets = {uid: t for uid, t in targets.items() if uid in obs.robots}
```

防止字典无限增长，并避免给已死单位保留旧角色。

## 4. Context 与 BFS

### Context（每回合重建，只读）

一个 frozen dataclass，对外暴露：

- 原始 `obs`、`config`、`mem` 引用
- `turn`、`south`、`north`、`width`、`me`（玩家索引）
- `walls`：合并永久记忆 + 本回合 `obs.walls` 的全图
- `crystals`、`mines`、`nodes`：按需合并
- `my_factory`、`my_units`、`enemy_units`、`enemy_factory`
- `danger_rows: set[int]`：未来 `SAFETY_HORIZON` 回合内会被扫掉的行集合（按当前 `southBound` 与滚动间隔斜坡公式预测）

### `passable` 谓词

两个变体：

- `passable_strict(cell, dir)` —— 墙检查 + `[south, north]` 范围 + `[0, width)` 范围 + 目标行不在 `danger_rows`。给工厂和高价值单位用。
- `passable_loose(cell, dir)` —— 同上但去掉 `danger_rows` 检查。给追求情报的 Scout 用。

### BFS

```python
def bfs(start, goal_predicate, *, passable_fn,
        occupied=frozenset(), max_dist=BFS_MAX_DIST) -> tuple[int, str] | None
```

- 标准 deque BFS，节点为 `(col, row)`。
- `goal_predicate(cell) -> bool`：调用方给谓词，可一次性查"最近 mining node"或"最近未知格"。
- `occupied`：可选不可踏入格集合（不含起点）。用于避开预订格。
- 返回 `(distance, first_step_dir)`，`first_step_dir ∈ {"NORTH","SOUTH","EAST","WEST"}`；起点已满足谓词时返回 `"IDLE"`。
- `max_dist` 步内找不到 → 返回 `None`。

每回合开销有界：每单位最多 1 次 BFS，单次 O(width × visible_rows) ≈ 600 节点，远低于 Kaggle 单步 1 秒上限。

## 5. 战略层 — 子策略

战略层按优先级应用 7 组子策略。下面编号与你确认过的清单对齐。

### 5.1 工厂保命（必选）

- **F1 安全边距**：工厂始终保持在 `south + SAFETY_MARGIN` 之上。低于此阈值时，本回合所有目标让位于工厂北上。
- **F2 JUMP 触发**：当北面有墙、2 回合内没有 SAPPER 能拆掉它、且 `jump_cd == 0` 时考虑 JUMP_NORTH。落点出界、落点处 1 步后会与敌方工厂相撞、或落入 `danger_rows` 且无后续可走，则取消 JUMP。
- **F3 造兵节流**：当 `factory_energy < factoryEnergy * LOW_ENERGY_RATIO` 时停止 BUILD_*，专注向北。

### 5.2 造兵决策（必选）

- **B1 造兵优先级序列**：第一台必造 SCOUT（最便宜、视野最大）。后续按需选：路径有墙瓶颈造 SAPPER；存在可达 mining node 且无在途 Miner 时造 HARVESTER；否则继续造 SCOUT 扩视野。
- **B2 mining node 触发造 Miner**：仅当某个 `mining_nodes` 项 BFS 距离 `< MINER_REACH_LIMIT` 且无在途 Miner 锁定它时造 Miner。
- **B3 拆墙瓶颈造 Worker**：当任意 EXPLORER 因非固定墙绕路代价 > `WALL_DETOUR_THRESHOLD` 时造 Worker。
- **B4 末期停造**：`episodeSteps - turn < LATE_GAME_STOP_BUILD` 时全部停造。

### 5.3 角色分配（必选）

- **R1** Scout → EXPLORER。
- **R2** Worker → 路径上有目标墙时为 SAPPER；否则为 GUARD（贴近工厂护送）。
- **R3** Miner → HARVESTER（锁定最近可达 node）。
- **R4** 角色每 `ROLE_REASSIGN_PERIOD` 回合或目标失效时重评估。其余时间角色保持粘性，避免来回抖动。

### 5.4 探索（建议）

- **E1 未知度评分**：对每个候选前沿格（与 `obs.walls == -1` 邻接的已知格），分数 = 以该格为中心的 `EXPLORE_KERNEL × EXPLORE_KERNEL` 窗口内 `-1` 单元数量。
- **E2 偏北偏置**：E1 分数乘以 `1 + NORTH_BIAS * (row - south) / (north - south + 1)`。
- **E3 中线门探测**：当中线镜像轴在记忆中显示为墙、但仍有未探测过的行时，派能量最高的 SCOUT 移到镜像轴的相邻格尝试 E/W 移动（不论是否有门，移动尝试都能更新墙图）。

### 5.5 能量经济（建议）

- **M1 TRANSFER 回流**：与工厂相邻且 `energy >= max_energy - TRANSFER_OVERFLOW_GAP` 的单位，或工厂能量低于 `LOW_ENERGY_RATIO * factoryEnergy` 时所有相邻单位，发起 `TRANSFER_*` 给工厂。
- **M2 crystal 顺路**：寻路过程中，若可见 crystal 在 BFS 路径绕路 ≤ `CRYSTAL_DETOUR_BUDGET` 步内，重定向走它。
- **M3 弃矿门槛**：与已知敌方工厂或 ≥ 2 个敌方单位的距离 ≤ `ENEMY_NEAR_RADIUS` 的 `mining_nodes` 项本回合不参与候选。

### 5.6 战斗与防御（可选）

- **C1 友军同格预防**：战术层按优先级预订下一回合落点；优先级低的单位若唯一安全动作会撞上更高优先级的友军，则改 IDLE。
- **C2 敌单位邻近警戒**：曼哈顿距离 ≤ `ENEMY_NEAR_RADIUS` 内有敌单位时，Scout 退向西北/东北；Worker 不再尝试在该敌人邻接格执行 BUILD_*。
- **C3 反 Scout 互撞**：我方 Scout 与敌方 Scout 邻接，且无更优 Scout 替代时，允许同归（1:1 兑子可接受）。
- **C4 工厂自保**：敌方工厂 `jump_cd <= 1` 且我方工厂在其 JUMP 范围内时，若侧向有安全格，本方工厂提前侧移。

### 5.7 后期收官（可选）

- **L1 能量囤积**：末 `LATE_GAME_HOARD` 回合内所有单位回流并相邻时 TRANSFER 给工厂。
- **L2 单位计数保留**：末期不派单位执行送死任务，活着的 IDLE 单位为 tiebreaker 计 1 个 unit。
- **L3 紧急 JUMP**：工厂距 `south` 仅 1 行且 NORTH 被堵时强制尝试 JUMP_NORTH（仍受游戏规则约束 —— `jump_cd == 0` 才合法；否则有 NORTH 走 NORTH）。

## 6. 战术层（Tactics）

### 6.1 调度顺序

每回合按 `(factory, miners, workers, scouts)` 顺序处理。每个单位决策完后把预测落点加入 `reservations`，后续单位需避开（除非能赢碰撞）。

```python
reservations: set[tuple[int,int]] = set()
for unit in sorted(my_units, key=type_priority):
    action = decide(unit, ctx, plan, reservations)
    actions[unit.uid] = action
    reservations.add(predict_next_cell(unit, action))
```

`predict_next_cell` 对应：`BUILD_*` → spawn 格；`JUMP_*` → 落点；移动 → 目标邻格；`IDLE`、`TRANSFORM`、`BUILD_DIR`、`REMOVE_DIR`、`TRANSFER_*` → 自身格。

### 6.2 单位决策流水线

```
1) 死亡过滤：枚举合法动作，剔除：
   - 踏出南/北界
   - 撞入 crush 等级更高的敌方单位
   - 撞入已被预订的更高 crush 等级友军
   - 撞入已被预订的同类型友军（互毁）
2) 角色专属意图：
   - FACTORY: F1 → F2 → B1..B4 → NORTH 或 IDLE
   - HARVESTER: 在 node 上则 TRANSFORM；否则 BFS 到指定 node
   - SAPPER: 已到目标墙则 BUILD_DIR/REMOVE_DIR；否则 BFS
   - EXPLORER: BFS 到指定前沿格
   - GUARD: BFS 到工厂护送格
3) 把意图转换为首步方向（BFS）或特殊动作。
4) 若意图首步被步骤 1 过滤掉：
   - 用方向打分（NORTH > E/W > SOUTH）重排幸存合法动作
   - 取最高分；都没有则 IDLE。
```

### 6.3 退化规则

流水线保证即使规划失败也产出合法动作：

- BFS 返回 None → 退化为方向打分；全堵 → IDLE。
- JUMP 落点全部不安全 → 取消 JUMP，下回合再试。
- Worker 的 BUILD/REMOVE 目标是固定墙（外圈或镜像轴）→ 跳过该指令，退化为 GUARD。
- Miner 到达目标但 node 已消失 → 退化为 GUARD。

### 6.4 工厂护送

工厂将要 NORTH 时，循环开始就把工厂目标格和它北侧第二格都加入 `reservations`，确保友军不挡路。

## 7. 可调常量

`main.py` 顶部 `# === TUNABLES ===` 块：

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

通过本地 self-play 调参，不依赖远程服务。

## 8. 测试策略

`tests/` 下三类轻量检查（不打入提交包）：

1. **冒烟测试（`tests/test_agent.py`）**：构造一个虚构 `obs`（一个工厂、几面墙、一个 crystal）。断言 `agent(obs, config)` 返回 dict，key 都是己方 UID，value 都是合法动作字符串集合内的元素。
2. **BFS 单元测试**：手工拼一个小墙图。断言有路时首步方向正确、全堵时返回 `None`。
3. **本地对战脚本**：用 `kaggle_environments.make("crawl")` 对 `random` 跑 5 个种子；胜率 ≥ 80% 通过。

不强制 80% 覆盖率 —— 仓库通用规则适用于 `src/` 布局的工程，本提交是单文件竞赛 agent，不适用。

## 9. 文件布局

```
main.py                  ← 提交文件，单文件
docs/superpowers/specs/  ← 本 spec 及后续 spec
tests/test_agent.py      ← 冒烟 + BFS 单元测试，不打包
```

提交命令：

```
kaggle competitions submit maze-crawler -f main.py -m "<message>"
```

## 10. 公开风险

- **中线门探测**：`doorProbability=0.08` 让门稀疏。E3 收益可能有限；若本地基准显示 vs 关闭 E3 收益 < 5%，删掉。
- **JUMP 安全评估**：1 步前瞻不能识别敌方两回合合围落点。在轻量预算下可接受。
- **Sapper 卡死**：BUILD/REMOVE 顺序错误针对固定墙时会以 100/回合白白扣能量。外圈与镜像轴的 `is_fixed_wall` 判断必须有单元测试覆盖。
