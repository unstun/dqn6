# DQN6/AGENTS.md（Ubuntu 24.04 + ros2py310）

> 作用域：本文件适用于 `/home/sun/phdproject/dqn/DQN6/**`。
> 通用环境/安装/成功口径优先参考上层 `../AGENTS.md`。

## 0. 硬约束（必须遵守）

1) 称呼：每次回复默认以“帅哥，”开头（除非你明确要求不需要）。
2) 先计划后动手：写/改任何文件前，必须先输出 3–7 步「实施计划」+「将改动的文件清单」+「风险点」+「验证方式」，并等待你回复“开始/按计划执行”后才允许实际修改。
3) 回复语言：默认中文。
4) 学术定义：DQN/DDQN/Double Q-learning 等术语与实现需符合原论文定义；不确定时先查证再实现。
5) 可复现：每次代码改动后，在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`（记录复现实验命令、seed、关键超参、变更摘要）；纯文档改动可豁免。
6) 文档同步：本目录的 `AGENTS.md` 与 `CLAUDE.md` 必须逐行一致；修改任一文件时同步修改另一份，并用 `diff -u AGENTS.md CLAUDE.md` 验证。

## 1. 最小自检 / 常用命令

- 自检：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 训练/推理（profile 配置在 `configs/*.json`）：
  - `conda run -n ros2py310 python train.py --profile <name>`
  - `conda run -n ros2py310 python infer.py --profile <name>`
- 输出目录：默认写入 `runs/`（可用 `--runs-root` 覆盖；见 `amr_dqn/cli/train.py`、`amr_dqn/cli/infer.py`）。
- 说明文档：`README.md`、`runtxt.md`。

## 2. 已知问题 / 踩坑（全部保留）

### 2.1 远端 SSH 执行注意事项（已踩坑）

- `conda run` 不会自动 cd 到项目目录，必须使用 `conda run --cwd <项目绝对路径>` 指定工作目录。
  - 正确：`ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/DQN6 -n ros2py310 python train.py ..."`
  - 错误：`ssh ubuntu-zt "conda run -n ros2py310 python train.py ..."`（会在 `~` 下找不到 `train.py`）
- 远端 `~/.bashrc` 的 conda init 块必须放在 interactive guard（`case $- in`）之前，否则 SSH 非交互式命令无法找到 conda。
  - 以往记录：2026-02-21 曾处理过该问题；若复现请优先检查 `~/.bashrc` 及备份 `~/.bashrc.bak.*`。

### 2.2 文件写入限制（Claude Code 客户端问题）

- 单次 Write / Edit 工具调用写入内容不得超过 50 行；超过时必须拆成多次调用（先写前 50 行，再追加后续内容）。
- 原因：Claude Code 客户端在单次写入过大时可能静默失败或进入死循环，导致工具调用反复失败。

### 2.3 联网调研注意事项（Claude Code 客户端问题）

- WebFetch 和 WebSearch 不混在同一批并行调用（WebFetch 403 会级联拖垮同批 WebSearch）。
- 每批并行最多 2 个同类调用；优先 arXiv / GitHub 等开放源。
- PDF 链接大概率解析失败，优先用 HTML 版本（如 `arxiv.org/html/`）。
- 付费墙站点（tandfonline / sciencedirect / springer）：WebFetch 可能 403，改用 Playwright：
  1. `browser_navigate` 打开 URL
  2. `browser_wait_for` 等 5 秒（Cloudflare 自动验证）
  3. `browser_snapshot` 获取页面内容

### 2.4 一整套 SSH 远端运行 + 本地回填工作流（推荐）

- 目标：训练/推理在远端执行，本地只同步结果到 `runs/`，保持“像本地运行一样”的目录结构，同时避免本地卡顿。
- 统一变量（示例）：
  - `REMOTE=ubuntu-zt`
  - `PROJ=/home/sun/phdproject/dqn/DQN6`
  - `ENV=ros2py310`
  - `PROFILE=forest_a_all6_300_cuda`
  - `EXP=forest_a_all6_300_cuda`

- 1) 远端后台启动（train -> infer 串行）：
  - `ssh $REMOTE 'set -euo pipefail; cd '"$PROJ"'; ts=$(date +%Y%m%d_%H%M%S); log="runs/'"$PROFILE"'_${ts}.log"; nohup bash -lc "conda run --cwd '"$PROJ"' -n '"$ENV"' python train.py --profile '"$PROFILE"' && conda run --cwd '"$PROJ"' -n '"$ENV"' python infer.py --profile '"$PROFILE"'" > "$log" 2>&1 & echo PID=$!; echo LOG=$log'`

- 2) 持续监控（建议每 30–90 秒一次）：
  - `ssh $REMOTE "ps -eo pid,ppid,stat,etime,cmd | grep -E 'python (train|infer)\\.py --profile $PROFILE' | grep -v grep"`
  - `ssh $REMOTE "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | grep python || true"`
  - `ssh $REMOTE "find $PROJ/runs/$EXP -maxdepth 2 -type d | sort | tail -n 20"`

- 3) 完成判定（以 all6 为例）：
  - 训练阶段应产出 6 个模型：`mlp-dqn.pt`、`mlp-ddqn.pt`、`mlp-pddqn.pt`、`cnn-dqn.pt`、`cnn-ddqn.pt`、`cnn-pddqn.pt`。
  - 推理阶段应在 `runs/$EXP/train_<timestamp>/infer/<timestamp>/` 下产出 KPI 与图表（如 `table2_kpis.csv`、`fig12_paths*.png`）。

- 4) 回填到本地 `runs/`（关键：不要放到临时下载目录）：
  - `rsync -av --partial $REMOTE:$PROJ/runs/$EXP/train_<timestamp>/ $PROJ/runs/$EXP/train_<timestamp>/`
  - `rsync -av $REMOTE:$PROJ/runs/<profile_log>.log $PROJ/runs/`
  - `rsync -av $REMOTE:$PROJ/configs/$PROFILE.json $PROJ/runs/$EXP/`
  - `printf '%s\n' 'train_<timestamp>' > $PROJ/runs/$EXP/latest.txt`

- 5) 回填后校验：
  - `ssh $REMOTE "find $PROJ/runs/$EXP/train_<timestamp> -type f | wc -l"`
  - `find $PROJ/runs/$EXP/train_<timestamp> -type f | wc -l`
  - `test -f $PROJ/runs/$EXP/train_<timestamp>/infer/<timestamp>/table2_kpis.csv && echo OK`
  - `cat $PROJ/runs/$EXP/latest.txt`（应指向本次 `train_<timestamp>`）
