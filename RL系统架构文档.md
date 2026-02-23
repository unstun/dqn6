# RL 系统架构文档

## 1. 文档范围

本文档覆盖 `DQN6` 仓库内基于 DQN 系列的强化学习（Reinforcement Learning）智能体架构与完整执行链路，不包含非 RL 基线算法的内部实现细节。

代码入口：

- `train.py` → `amr_dqn/cli/train.py`（训练）
- `infer.py` → `amr_dqn/cli/infer.py`（推理）

核心模块：

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| 智能体 | `amr_dqn/agents.py` | RL 智能体定义、时序差分（TD）更新、模型保存/加载 |
| 神经网络 | `amr_dqn/networks.py` | MLP / CNN Q 网络 |
| 经验回放 | `amr_dqn/replay_buffer.py` | 经验回放缓冲区（含示范保护） |
| 环境 | `amr_dqn/env.py` | `AMRGridEnv` / `AMRBicycleEnv` 及动作可行性掩码 |
| 地图 | `amr_dqn/maps/` | 地图注册、程序化森林地图生成 |
| 基线 | `amr_dqn/baselines/pathplan.py` | Hybrid A*、RRT* 封装 |
| 指标 | `amr_dqn/metrics.py` | 路径质量关键性能指标（KPI）计算 |
| 调度 | `amr_dqn/schedules.py` | ε-贪心探索率衰减策略 |
| 平滑 | `amr_dqn/smoothing.py` | Chaikin 角切割路径平滑 |
| 实验管理 | `amr_dqn/runs.py` | 实验目录创建与解析 |
| 运行时 | `amr_dqn/runtime.py` | PyTorch / CUDA 设备选择 |
| 配置 | `amr_dqn/config_io.py` | JSON 配置加载与合并 |

---

## 2. RL 智能体全集

以 `amr_dqn/agents.py::parse_rl_algo` 为唯一规范，当前仓库共 6 个规范智能体：

1. `mlp-dqn` — MLP 骨干 + 标准 DQN
2. `mlp-ddqn` — MLP 骨干 + Double DQN
3. `mlp-pddqn` — MLP 骨干 + Double DQN + Polyak 软更新
4. `cnn-dqn` — CNN 骨干 + 标准 DQN
5. `cnn-ddqn` — CNN 骨干 + Double DQN
6. `cnn-pddqn` — CNN 骨干 + Double DQN + Polyak 软更新

历史别名（Legacy Alias）：

| 别名 | 映射到 |
| --- | --- |
| `dqn` | `mlp-dqn` |
| `ddqn` | `mlp-ddqn` |
| `iddqn` | `mlp-pddqn` |
| `cnn-iddqn` | `cnn-pddqn` |

> 说明：`mlp-pddqn` / `cnn-pddqn` 是本仓库定义的"Double DQN + Polyak 软目标更新"变体，并非外部论文中的独立 IDDQN 算法族。

---

## 3. 智能体能力矩阵

| 规范名称 | 骨干网络 | TD 目标计算 | 目标网络更新方式 | 默认超参数 |
| --- | --- | --- | --- | --- |
| `mlp-dqn` | `MLPQNetwork` | DQN（目标网络取 max） | 硬更新 | `eps_start=0.6`, `n_step=3`, `tau=0` |
| `mlp-ddqn` | `MLPQNetwork` | Double DQN（在线网络 argmax + 目标网络评估） | 硬更新 | 同上 |
| `mlp-pddqn` | `MLPQNetwork` | Double DQN | Polyak 软更新 | `eps_start=0.6`, `n_step=3`, `tau=0.01` |
| `cnn-dqn` | `CNNQNetwork` | DQN | 硬更新 | 同 `mlp-dqn` |
| `cnn-ddqn` | `CNNQNetwork` | Double DQN | 硬更新 | 同 `mlp-ddqn` |
| `cnn-pddqn` | `CNNQNetwork` | Double DQN | Polyak 软更新 | 同 `mlp-pddqn` |

共享配置来自 `AgentConfig`：`gamma=0.995`、`batch_size=128`、`learning_rate=5e-4`、`replay_capacity=100000`。

---

## 4. 神经网络架构（`amr_dqn/networks.py`）

### 4.1 MLP Q 网络（`MLPQNetwork`）

- 全连接网络，默认 3 层隐藏层 × 256 单元，ReLU 激活
- 输入：扁平化观测向量
- 输出：所有离散动作对应的 Q 值

### 4.2 CNN Q 网络（`CNNQNetwork`）

CNN + MLP 混合架构，根据观测维度自动推断输入布局：

| 环境 | 标量维度 | 地图通道数 | 地图尺寸 |
| --- | --- | --- | --- |
| `AMRGridEnv` | 5 | 1（占据栅格） | N×N |
| `AMRBicycleEnv` | 11 | 2（占据栅格 + 到达代价图） | N×N |

- CNN 分支：3 层卷积（32→64→64 通道）
- 将 CNN 特征与标量特征拼接后送入全连接头（2 层 × 256 单元）
- 布局推断函数：`infer_flat_obs_cnn_layout()`

---

## 5. 经验回放缓冲区（`amr_dqn/replay_buffer.py`）

`ReplayBuffer` 类，固定容量环形缓冲区。

存储字段：`(obs, action, reward, next_obs, done, next_action_mask, demo, n_steps)`

核心特性：

1. 示范保护：标记为 `demo=True` 的转移不会被非示范数据覆盖（DQfD 风格）
2. 动作掩码存储：保存下一状态的可行动作掩码，供 TD 目标计算时屏蔽非法动作
3. n 步回报：记录每条转移的实际展开步数

---

## 6. 环境与输入输出接口（`amr_dqn/env.py`）

所有 RL 智能体共用以下两个环境，均遵循 OpenAI Gym 接口。

### 6.1 观测空间

**`AMRGridEnv`（简单栅格世界）**

- 维度：`5 + obs_map_size²`
- 内容：智能体/目标位置标量（归一化至 [-1, 1]）+ 降采样占据栅格（默认 12×12）

**`AMRBicycleEnv`（自行车运动学模型，森林场景）**

- 维度：`11 + 2 × obs_map_size²`
- 内容：位姿（x, y, θ）、速度、转向角、目标（x, y, θ）、距离等标量 + 降采样占据栅格 + 降采样到达代价图
- 可选：36 扇区激光雷达（10° 分辨率）

### 6.2 动作空间

| 环境 | 动作类型 | 动作数 | 说明 |
| --- | --- | --- | --- |
| `AMRGridEnv` | 离散 8 连通方向 | 8 | 上下左右 + 四个对角 |
| `AMRBicycleEnv` | 离散组合动作 | 35 | 7 个转向角速率 × 5 个加速度（阿克曼运动学） |

### 6.3 自行车模型参数

- 轴距：0.6 m
- 最大速度：2.0 m/s
- 最大转向角：27°
- 积分方式：欧拉法，dt = 0.05 s
- 碰撞检测：双圆轮廓近似（半径 0.436 m，中心距后轴 ±0.231 m）

---

## 7. 安全性与可行性约束机制

以下机制主要作用于 `AMRBicycleEnv`（森林场景），核心目标：避免碰撞、避免安全但无进展的卡死状态，并在无可行动作时提供兜底策略。

### 7.1 约束判定（环境层，`amr_dqn/env.py`）

#### 几何安全性

- 车辆碰撞由双圆轮廓 + 欧氏距离变换（EDT）间距判定
- 动作评估采用短视野恒定动作轨迹展开（非单步判定）

#### 单动作可行性 `is_action_admissible(...)`

- 基于当前位姿代价 `cost0` 与展开终点代价 `cost1`
- 必须满足：展开期间无碰撞；最小障碍距离 ≥ 阈值；且到达目标或代价降低 ≥ `min_progress_m`
- 倒车放行为条件性的：仅当无任何前进可行动作时，才允许满足阈值的倒车动作

#### 多动作掩码 `admissible_action_mask(...)`

- 并行评估 35 个离散动作，生成布尔掩码
- 基础掩码：安全 ∧ 有进展
- 若 `allow_reverse=True` 且无进展动作 → 退化为安全倒车动作
- 若 `fallback_to_safe=True` 且仍为空 → 退化为仅安全动作（不要求进展）

#### 最终兜底 `_fallback_action_short_rollout(...)`

- 优先选安全且代价降低最优的动作（同代价下优先更大间距）
- 若仍无可选 → 退化到单步后间距最大的动作

### 7.2 训练阶段约束接入（`amr_dqn/cli/train.py`）

1. `--forest-action-shield`（默认开启）
   - 每步维护可行动作掩码：`admissible_action_mask(h=6, min_od=0, min_progress=1e-4, fallback_to_safe=True)`
   - 智能体通过 `act_masked(...)` 在掩码内做 ε-贪心采样
2. `--forest-expert-exploration`（默认开启）
   - 训练早期按概率混入专家策略
   - 非专家分支且未启用动作屏蔽时，仍会做一次 `is_action_admissible` 检查与兜底链
3. 经验回放中保存 `next_action_mask`
   - `agent.observe(...)` 写入下一状态掩码，TD 目标的 next-Q 计算会对非法动作做掩码
4. 训练内贪心评估（选最优检查点）也走可行性门控
   - 使用 `horizon_steps=15` 的可行性检查与兜底，减少训练-推理不一致

### 7.3 推理阶段约束接入（`amr_dqn/cli/infer.py`）

`rollout_agent(...)` 在森林场景下并非直接取 `argmax(Q)`，而是分层决策：

1. 取 `a0 = argmax(Q)`
2. 检查 `a0` 是否可行（默认：`h=15`, `min_od=0.0`, `min_progress=1e-4`）
3. 若不可行：
   - 在 top-k（默认 `k=10`）中找第一个可行动作
   - 若无，则用 `admissible_action_mask(..., fallback_to_safe=False)` 做仅进展掩码
   - 若仍为空，调用 `_fallback_action_short_rollout(...)`

结论：推理时策略为"Q 值主导 + 可行性门控"，而非原始网络输出。

### 7.4 推理阶段参数入口与默认值

| CLI 参数 | 默认值 | `rollout_agent` 对应参数 |
| --- | --- | --- |
| `--forest-adm-horizon` | `15` | `forest_adm_horizon` |
| `--forest-topk` | `10` | `forest_topk` |
| `--forest-min-progress-m` | `1e-4` | `forest_min_progress_m` |
| `--forest-min-od-m` | `0.0` | `forest_min_od_m` |

### 7.5 边界与风险

1. 上述约束机制主要针对 `AMRBicycleEnv`；`AMRGridEnv` 不走同一套可行性轨迹展开判定
2. 若训练时同时关闭 `forest_action_shield` 和 `forest_expert_exploration`，训练采样可能退化为较弱约束（但推理仍会做门控）
3. 约束过强会牺牲探索，过弱会提升碰撞/卡死风险，需结合 KPI（成功率、规划代价、综合评分）调参

---

## 8. 训练架构（`amr_dqn/cli/train.py`）

### 8.1 入口与算法规范化

1. 解析 `--profile` / `--config` 加载 JSON 配置
2. `--rl-algos` 统一规范化（`all` 展开为全部 6 个）
3. 创建实验目录：`runs/<experiment>/train_<timestamp>/`

### 8.2 环境构建

1. 森林场景使用 `AMRBicycleEnv`
2. 非森林场景使用 `AMRGridEnv`
3. 可配置随机起点/终点、课程学习等

### 8.3 智能体创建与算法分发

1. 每个算法创建一个 `DQNFamilyAgent` 实例
2. `parse_rl_algo` 决定：
   - `arch`（MLP / CNN）
   - `base_algo`（DQN / DDQN）
3. 三类配置分支：DQN、DDQN、PDDQN（仅 `target_update_tau > 0` 时）

### 8.4 训练循环

1. 动作采样：`act` 或 `act_masked`（含 ε-贪心）
2. 森林场景可选增强：
   - 示范预填充（专家示范预填充经验回放缓冲区，目标约 20k 条转移）
   - 示范预训练（`pretrain_on_demos`，50k 步监督预训练，含行为克隆 + 间隔损失）
   - 专家探索（训练期混入专家策略，70%→0%，覆盖前 60% 回合）
   - 动作屏蔽（可行动作掩码）
   - 课程学习（起点逐步远离目标）
3. 经验入库：`ReplayBuffer.add`（含 `next_action_mask`、`demo`、`n_steps`）
4. 周期性更新：`agent.update()`

### 8.5 TD 更新逻辑

1. DQN：`target = r + γⁿ · max_a' Q_target(s', a')`
2. DDQN / PDDQN：`a* = argmax Q_online(s')`，再用 `Q_target(s', a*)` 评估
3. 示范损失（DQfD 风格）：
   - 间隔损失（Large-margin Loss）
   - 交叉熵损失（行为克隆）

### 8.6 检查点选择与保存

1. 训练后对"最终策略"、"最佳回合策略"、"预训练后策略"三个候选做贪心评估，选最优
2. 模型保存路径：`runs/<exp>/train_<ts>/models/<env>/<canonical_algo>.pt`
3. 训练输出文件：
   - `training_returns.csv`（回合回报）
   - `training_eval.csv` / `.xlsx`（评估指标）
   - `fig13_rewards.png`（训练曲线）

---

## 9. 推理与评估架构（`amr_dqn/cli/infer.py`）

### 9.1 算法与模型加载

1. `--rl-algos` 再次规范化
2. 按 `<models_dir>/<env>/<algo>.pt` 载入
3. 兼容历史命名：
   - `mlp-pddqn` 可回退查找 `iddqn.pt`
   - `cnn-pddqn` 可回退查找 `cnn-iddqn.pt`

### 9.2 RL 轨迹展开

`rollout_agent` 主流程：

1. 计算 Q 值并取贪心动作
2. 森林场景做可行性检查（详见第 7.3 节）
3. 若当前动作不可行 → top-k 查找 → 进展掩码 → 短视野兜底

### 9.3 评估公平性

启用随机起点/终点时：

1. 先采样固定的一组 `(start, goal)`
2. 同一组样本用于所有 RL 智能体与基线
3. 横向比较在同分布、同样本上进行

### 9.4 关键性能指标与结果文件

输出目录：`infer/<timestamp>/`

| 文件 | 内容 |
| --- | --- |
| `table2_kpis.csv` | 每次运行的成功率与 KPI |
| `table2_kpis_mean.csv` | 均值汇总表 |
| `*_raw.csv` | 原始数值 |
| `fig12_paths.png` | 路径可视化 |
| `fig13_controls.png` | 控制量可视化 |

关键指标：

1. 成功率（`success_rate`）
2. 规划代价（`planning_cost`）— 受成功率惩罚
3. 综合评分（`composite_score`）— 受成功率惩罚

其他指标：路径长度（m）、路径时间（s）、平均曲率（1/m）、规划耗时（s）、拐角数量、最大拐角角度（°）。

---

## 10. 地图系统（`amr_dqn/maps/`）

### 10.1 地图接口

- `MapSpec` 协议：所有地图的统一接口
- `ArrayGridMapSpec`：基于 NumPy 数组的栅格地图

### 10.2 程序化森林地图生成（`maps/forest.py`）

使用泊松圆盘采样（Poisson Disk Sampling）放置树木，支持配置树干数量、间隙大小、密度，并验证自行车运动学可达性。

| 地图 | 尺寸（格） | 间隙（m） | 树木数 | 分辨率 |
| --- | --- | --- | --- | --- |
| `forest_a` | 360×360 | 3.0 | 85 | 0.1 m/格 |
| `forest_b` | 96×96 | 1.35 | 28 | 0.1 m/格 |
| `forest_c` | 160×160 | 1.25 | 85 | 0.1 m/格 |
| `forest_d` | 96×96 | 1.30 | 28 | 0.1 m/格 |

生成的地图缓存于 `_FOREST_CACHE`，预计算地图存放在 `maps/precomputed/`。

---

## 11. 森林场景训练稳定化技术

森林环境的长视野学习采用了以下稳定化手段：

| 技术 | 说明 |
| --- | --- |
| 示范预填充 | 用专家轨迹（Hybrid A* 或到达代价梯度）预填充约 20k 条转移 |
| 监督预训练 | 50k 步行为克隆 + 间隔损失，早停条件为贪心策略到达目标 |
| 专家探索 | 训练期混入专家动作（70%→0%），覆盖前 60% 回合 |
| 动作屏蔽 | 训练时用可行动作掩码过滤不安全/无进展动作 |
| 课程学习 | 起点从目标附近逐步扩展到全距离 |
| DQfD 损失 | 间隔损失 + 交叉熵损失作用于示范转移 |
| 示范保护 | 经验回放缓冲区保护示范转移不被覆盖 |
| n 步回报 | 默认 n=3，跨更长视野做自举 |
| 高折扣因子 | γ=0.995（常规为 0.99），适应长回合 |
| 检查点选择 | 评估多个候选（最终、最佳回合、预训练后），取最优 |

---

## 12. 配置系统

配置文件位于 `configs/*.json`，结构为 `{"train": {...}, "infer": {...}}`。

加载优先级：

1. `--config <path>`：直接指定路径
2. `--profile <name>`：查找 `configs/<name>.json`
3. 默认：`configs/config.json`

CLI 参数可覆盖 JSON 中的任何值。

---

## 13. 非 RL 基线边界说明

下列算法在本仓库评测中属于基线，不是 RL 智能体：

1. Hybrid A*（混合 A* 运动规划）
2. RRT*（快速随机树优化）

它们在 `infer.py` 中可与 RL 结果同表展示，但不属于第 2 节的 RL 智能体规范列表。封装位于 `amr_dqn/baselines/pathplan.py`。

---

## 14. 组件关系总览

```text
┌──────────────────────────────────────────────────────────┐
│                      训练流水线                           │
│  amr_dqn/cli/train.py::train_one                         │
└────────────┬─────────────────────────────────────────────┘
             │
             ├─→ DQNFamilyAgent (agents.py)
             │   ├─→ MLPQNetwork / CNNQNetwork (networks.py)
             │   ├─→ ReplayBuffer (replay_buffer.py)
             │   └─→ linear_epsilon (schedules.py)
             │
             ├─→ AMRBicycleEnv / AMRGridEnv (env.py)
             │   ├─→ MapSpec (maps/__init__.py)
             │   │   └─→ generate_forest_grid (maps/forest.py)
             │   └─→ 专家动作 (hybrid_astar / cost_to_go)
             │
             └─→ collect_forest_demos()

┌──────────────────────────────────────────────────────────┐
│                      推理流水线                           │
│  amr_dqn/cli/infer.py::infer_one_rl                      │
└────────────┬─────────────────────────────────────────────┘
             │
             ├─→ DQNFamilyAgent.load() → act() / act_masked()
             ├─→ AMRBicycleEnv → admissible_action_mask()
             ├─→ 基线 (baselines/pathplan.py)
             └─→ 指标计算 (metrics.py)
```

---

## 15. 快速核对清单（防漏项）

若后续改代码，请按以下顺序核对是否新增/变更 RL 智能体：

1. `amr_dqn/agents.py::parse_rl_algo` 是否新增规范名或别名
2. `amr_dqn/cli/train.py` 中 `canonical_all`、`algo_cfgs`、`algo_labels` 是否同步
3. `amr_dqn/cli/infer.py` 中 `canonical_all`、`algo_label`、`resolve_model_path` 兼容表是否同步
4. `configs/*.json` 中 `rl_algos` 是否引用了新名字
5. 输出检查点命名是否仍为 `<canonical_algo>.pt`

---

## 16. 审查结论

基于当前 `DQN6` 代码树，RL 智能体只有且仅有 6 个规范项（第 2 节），未发现 SAC / PPO / TD3 等其它 RL 智能体实现入口。

本文档已覆盖：

1. 全部规范智能体与历史别名
2. 神经网络架构与经验回放机制
3. 环境接口、动作空间与安全约束
4. 训练、推理、模型保存、KPI 计算全链路
5. 地图系统与森林场景稳定化技术
6. 配置系统与非 RL 基线边界
