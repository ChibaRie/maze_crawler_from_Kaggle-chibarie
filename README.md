# Maze Crawler AI Agent

Kaggle [Maze Crawler](https://www.kaggle.com/competitions/maze-crawler/) 竞赛的 AI Agent，基于启发式状态机 + BFS 寻路架构。

## 简介

Maze Crawler 是一场双人即时策略对抗赛。双方各持一座 **Factory**（工厂），在一座 20x20、持续向北卷动的迷宫中生存。地图具有东西对称性，存在战争迷雾。最后存活的工厂获胜；若双方均存活至 500 步，则按 **总能量 → 单位数量 → 平局** 依次决胜。

**核心机制：**
- 四种机器人：Factory（不可摧毁、可建造/跳跃）、Scout（廉价侦察、视野5）、Worker（建/拆墙）、Miner（在矿点变为能量矿）
- 碾压等级：Factory > Miner > Worker > Scout，**友军误伤同样生效**
- 地图向南卷动，速度从每4回合一次线性加速至每回合一次（步400后），超出南边界的一切被销毁

## 项目结构

```
.
├── main.py                 # 单文件提交 —— 完整 Agent 实现
├── tests/
│   ├── conftest.py         # 共享测试 fixtures
│   ├── test_smoke.py       # 冒烟测试（基本管线验证）
│   ├── test_memory.py      # 记忆层单元测试
│   ├── test_passable.py    # 通行性判断测试
│   ├── test_bfs.py         # BFS 寻路测试
│   ├── test_tactics.py     # 策略与战术层测试
│   └── match_random.py     # 对战随机 Agent 的手动测试工具
├── docs/
│   └── superpowers/
│       ├── specs/           # 设计文档
│       └── plans/           # 实现计划
├── README.md               # 竞赛规则速查
├── AGENTS.md               # 入门指南与 CLI 操作手册
└── overview_and_dataset.md  # 竞赛官方介绍
```

## 设计思路

### 总体架构：四层流水线

```
每回合:  Memory → Context → Strategy → Tactics → Actions
```

| 层 | 职责 | 关键数据 |
|----|------|----------|
| **Memory** | 跨回合持久状态 | 已知墙体、矿点、己方工厂历史位置 |
| **Context** | 单回合只读快照 | 解析后的单位列表、晶体/矿位置、危险行 |
| **Strategy** | 角色分配 + 建造决策 | 角色（EXPLORER/HARVESTER/SAPPER/GUARD）、建造优先级 |
| **Tactics** | 每单位行动选择 | BFS 寻路、碾压过滤、反震荡、晶体拾取 |

### 为什么选这个架构？

竞赛限制单文件提交且需要低延迟决策。我们放弃了 MCTS/深度搜索等计算密集型方案，选择了 **启发式状态机 + BFS** 的轻量路线——在 Kaggle 的运行环境中，每回合决策需在毫秒级完成。

四层流水线的设计保证了：
- **Memory 层**的持久化与剪枝解决了战争迷雾下的信息不完全问题
- **Context 层**的 frozen dataclass 避免了可变状态带来的 bug
- **Strategy 层**的角色系统让多单位协作不混乱
- **Tactics 层**的碾压过滤 + 预约机制防止友军误伤

---

## 开发历程

### v1：从零构建（Kaggle 得分 554.6）

**第一步：理解赛题**

通读竞赛规则，提炼出关键约束：
- 工厂存活是唯一硬性胜利条件 → **工厂安全是第一优先级**
- 能量是决胜平局的 tiebreaker → **不能乱花能量**
- 南边界加速卷动 → **必须持续向北移动**
- 友军碾压同样致命 → **需要防碰撞机制**

**第二步：分层设计**

采用自底向上的开发顺序：

1. **基础设施**：墙壁位域常量、方向偏移表、模块级 `_MEM` 字典
2. **Memory 层**：`memory_update()` 负责墙壁/矿点/矿节点的跨回合合并，以及南边界以下条目的剪枝
3. **Context 层**：`build_context()` 将原始 obs 解析为结构化的 `Context` dataclass，包含 `Unit` 对象列表、危险行集合等
4. **BFS 寻路**：`bfs()` 接受目标谓词 + 通行函数，返回 `(距离, 首步方向)`
5. **Strategy 层**：
   - `assign_roles()`：按单位类型和地图状态分配角色（EXPLORER/HARVESTER/SAPPER/GUARD）
   - `pick_factory_build()`：建造决策阶梯（先侦察兵 → 遇墙造工兵 → 有矿造矿工）
   - `assign_targets()`：按角色计算每个单位的目标格
6. **Tactics 层**：
   - `death_filter()`：过滤会导致被碾压或走出地图的动作
   - `decide_unit()`：综合角色、目标、BFS、碾压过滤生成最终动作
7. **Conductor**：`agent()` 函数串联四层管线，按类型优先级排序，使用预约集合防止友军碰撞

**第三步：测试驱动**

为每一层编写了单元测试（共 43 个），确保：
- 墙壁解析和坐标转换正确
- BFS 能绕过障碍物和已占格
- 碾压过滤遵循等级规则
- 工厂建造在能量/冷却/晚期各条件下的行为符合预期
- 整体管线输出合法动作

**v1 问题诊断：**

提交后得分 554.6，通过对战随机 Agent 达到 80% 胜率（4/5 种子）。分析失败场景发现：
- 工厂经常走入迷雾（未知区域）导致向南误入死路
- 建造单位太激进，能量被快速消耗
- 工厂在死胡同中来回震荡，浪费回合
- 没有拾取路上的晶体，白白损失能量

---

### v2：针对性改进（Kaggle 得分 884.6）

通过分析 v1 的失败模式和研究高分方案的设计思路，提炼出 **7 项关键改进**：

#### 改进 1：翻转规划顺序

```
v1:  Factory(0) → Miner(1) → Worker(2) → Scout(3)  ❌
v2:  Scout(0)   → Worker(1) → Miner(2) → Factory(3) ✅
```

**原因：** Scout 每回合都能移动，最先规划意味着它们会先承诺移动方向。当 Factory 最后规划时，它已经知道所有友军的去向，不会浪费 JUMP 冷却来躲自己的侦察兵。

#### 改进 2：大幅提高建造门槛

| 单位 | v1 门槛 | v2 门槛 | 差距 |
|------|---------|---------|------|
| Scout | 50 (成本价) | 650 (成本+600) | +600 |
| Miner | 300 | 800 (成本+500) | +500 |
| Worker | 200 | 700 (成本+500) | +500 |
| 2nd Scout | 50 | 850 (成本+800) | +800 |

**原因：** 能量是决胜 tiebreaker。v1 像流水一样花能量建单位，但多一个 Scout 的边际收益远不如保住 600 能量重要。

#### 改进 3：工厂使用已知格 BFS（拒绝穿越迷雾）

新增 `passable_known()` 函数：起点和终点都必须是已探索过的格子。

```python
def passable_known(ctx, cell, direction):
    if cell not in ctx.walls:       # 起点未知 → 拒绝
        return False
    if _wall_between(ctx, cell, direction):
        return False
    nxt = _step(cell, direction)
    return nxt in ctx.walls         # 终点未知 → 拒绝
```

**原因：** v1 的乐观穿越（未知视为可通行）会让 BFS 找到"穿过迷雾 7 步到达"的虚假路径，经常引导工厂向南走入死路。

#### 改进 4：工厂反震荡 + 死胡同跳跃逃生

- 在 `_MEM` 中追踪 `last_factory_pos`，避免折返到上一个位置
- 追踪 `factory_stuck_count`，当所有方向都不可走时使用 `JUMP_NORTH` 跳过墙壁
- v1 只在"靠近南边界且北面有墙"时才跳，v2 在任何卡住的情况下都会尝试跳跃

#### 改进 5：晶体拾取

为 Scout 和 Worker 添加晶体绕路逻辑：当单位能量未满时，BFS 搜索 12 步以内的最近晶体。

```python
if unit.type in (TYPE_SCOUT, TYPE_WORKER) and ctx.crystals:
    cap = _max_energy_for(ctx, unit)
    if cap is not None and (cap - unit.energy) > 5:
        # BFS 到最近晶体，最远 12 步
```

**原因：** 地面上的晶体是免费能量（10-50），不拾取白白浪费。

#### 改进 6：生成格安全检查

建造前检查北面一格：无墙阻隔、未被占据、未被预约。

```python
def _spawn_cell_clear(ctx, factory, occupied_cells):
    spawn = (factory.col, factory.row + 1)
    if spawn[1] > ctx.north: return False
    if _wall_between(ctx, factory.cell, "NORTH"): return False
    if spawn in occupied_cells: return False
    return True
```

同时改为条件性建造 Worker：只在附近检测到阻塞墙壁时才建。

#### 改进 7：矿节点生命周期管理

v1 只添加不删除矿节点；v2 在矿节点被占用（变为矿井）后从记忆中移除，避免 Miner 走向一个已经不存在的节点。

同时移除了工厂护送格预约（v1 中 `escort_cell` 锁定工厂北侧一格），因为在新的"Scout 先规划"顺序下这会不必要地锁住友军。

---

### v2 效果

| 指标 | v1 | v2 |
|------|----|----|
| Kaggle 得分 | 554.6 | **884.6** |
| 对战随机 Agent 胜率 | 80% (4/5) | **100% (5/5)** |
| 测试用例 | 43 通过 | 43 通过 |

---

## 关键算法

### BFS 寻路

标准广度优先搜索，支持：
- 自定义**目标谓词**（如"是晶体格"、"是矿节点"）
- 可插拔的**通行函数**（loose/strict/known-only）
- **已占格回避**（防止 BFS 穿过友军）
- **最大距离限制**（避免搜索整张地图）

```python
def bfs(start, goal_predicate, *, passable_fn, occupied, max_dist):
    → (distance, first_step_direction) or None
```

### 碾压过滤 (Death Filter)

在候选动作列表中过滤掉必死选项：
- 走出地图边界
- 走入高碾压等级的敌方单位所在格
- 走入被同级或更高级友军预约的格

### 角色系统

每个非工厂单位被分配一个角色，决定其目标和行为：

| 角色 | 分配条件 | 行为 |
|------|----------|------|
| EXPLORER | Scout 默认 | BFS 到未探索边界（frontier score 最高处） |
| HARVESTER | Miner + 有矿节点 | BFS 到最近矿节点 → TRANSFORM |
| SAPPER | Worker + 有阻塞墙 | BFS 到墙壁 → REMOVE |
| GUARD | 兜底角色 | 跟随工厂 |

角色具有粘性（`ROLE_REASSIGN_PERIOD = 8` 回合重新评估），避免频繁切换导致的振荡。

---

## 本地测试

```bash
# 安装依赖
pip install kaggle-environments pytest

# 运行单元测试
pytest tests/ -v

# 对战随机 Agent（5 个种子）
python tests/match_random.py

# 在 notebook 中可视化
python -c "
from kaggle_environments import make
env = make('crawl', configuration={'randomSeed': 42}, debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
```

## 后续改进方向

以下是 v1 设计文档中延迟的子策略，可能在 v3 中实现：

- **E3 中线门探测**：主动搜索连接东西半区的门
- **M3 弃矿阈值**：当矿太靠近南边界时放弃前往
- **C2-C4 战斗启发式**：主动碾压弱敌、引诱敌人到危险行
- **L1-L3 晚期战术**：晚期停止建造、回收单位能量、向工厂集结

## License

MIT
