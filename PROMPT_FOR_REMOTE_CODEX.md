# Prompt for the Codex instance on the Pro 6000 machine

请在这台 GPU 机器上继续 `PathRel` 研究。你收到的是一个独立的研究包，不要把它重新接回
`vi/`、ROS 或 G1 控制链；当前目标是先验证科学问题，而不是做工程拼接。

## 研究问题

给定不完整的 BEV 观测 `X`，学习空间相关的随机占据后验，并校准

`q(s,g,r|X) = P(存在一条能容纳 footprint 半径 r 的四邻域路径 | X)`。

论文的潜在贡献是：voxel 边际概率/ECE 不能决定非局部的路径可靠性；模型必须学习联合空间
后验，并用 task-level proper score 校准路径事件。当前实现只覆盖二值 latent world
(`TRAVERSABLE/BLOCKED`)、圆形 footprint 和 2.5-D 栅格拓扑，不要把它描述成完整动力学安全保证。

## 第一轮必须做的事

1. 阅读 `README.md`、`ROADMAP_ZH.md`、`ALGORITHM.md` 和全部 `src/pathrel`/`tests`。
2. 运行 `nvidia-smi`、`python scripts/check_environment.py`，确认使用 CUDA PyTorch；不要复制或
   修改旧环境。记录 GPU、驱动、Torch、CUDA、显存和 commit 状态。
3. 设置 `PYTHONPATH=src`，运行测试，确认 `skipped=0`；再运行最小 GPU smoke。若失败，先
   修复可复现的环境/shape/dtype 问题，并写入 `results/`，不要静默绕过测试。
4. 检查训练脚本的显存和耗时。可以添加 AMP、sample/query 分块或 profiler，但保持
   reachability 的标签、hard event 语义和 fp32 几何逻辑正确；不要为了速度把 event 概率改成
   fuzzy max-min 分数。

## P0：在接入真实数据前完成 death test

实现一个可复现的评估脚本（固定 seed、保存 JSON/CSV 和图），至少比较：

- per-query/radius constant predictor；
- 与 PathRel 相同 voxel 边际的 independent-Bernoulli sampler；
- direct `q(s,g,r)` MLP；
- connectivity-loss / edge-connectivity predictor（例如 promising-region connectivity 类方法）；
- 随机 occupancy completion 基线（优先 SCOPE；若成本过高，至少复用其 decoder/公开 OGM 设定）；
- FlatLands 的 stochastic/flow completion baseline；优先复用其 partial/full/valid mask 和
  官方 split，在其样本上离线计算 footprint-conditioned connectivity。
- deterministic occupancy + threshold；
- PathRel 去掉 reachability loss 的 ablation；
- 完整 PathRel。

扩展 synthetic generator，至少提供两个可见 context family，使隐藏门洞先验明显不同（例如
`P(open)=0.2` 与 `P(open)=0.8`），同时每个 context 重复多个隐藏世界。还要保留“相同边际、
不同联合拓扑”的冲突集。训练/验证按场景模板拆分，禁止相邻帧或同一模板泄漏。

报告：reachability Brier、NLL、ECE、可靠性图、false-safe rate、按 footprint 半径的曲线、
地图 marginal 指标，以及样本中门洞整体开/关的 joint-frequency。与 constant baseline 比较，
不要只报告训练 loss。若 independent-cell 或 direct-query 在相同校准和跨模板测试下追平，立即
把方向标记为失败，不继续堆网络或传感器。

## P1/P2：只有 P0 通过才做真实数据

- 先审计 ORFD：unknown/free 语义、累计 LiDAR 真值、断连和窄瓶颈 query 比例；不满足条件就
  不把它当主数据。
- 主实验优先审计 FlatLands：它已经提供 partial observation、多个合法 full layouts、valid
  mask 和 stochastic completion benchmark。先问“现有 completion samples 的 path-existence
  event 是否校准”，不要重复做一个 floormap completion 网络。若室内 floor-map 与小车越野
  支撑面不匹配，再转 UnScenes3D（occupancy + road elevation）和 WildOcc 做 2.5-D 验证。
  固定同一个公开/标准 BEV encoder，方法差异只放在后验与可靠性层。
- 使用 FlatLands 前先统计自然 query 的 reachable/unreachable、瓶颈、替代路径和 footprint
  半径分布；若负例或窄瓶颈少于约 10--15%，它只能作为 completion/OOD 基线，不能冒充主事件
  数据。query 采样只能依赖测试时可见的 observation/valid mask/goal，禁止用 GT 挑难例。
- related work/baseline 还要覆盖 SCOPE、diffusion-based occupancy completion 和 FlatLands；
  PathRel 的卖点必须是 event-level calibration，而不是“能生成可能地图”。FlatLands 若已
  覆盖 completion 的 sample/energy-score，就不能再把“partial-view 多世界生成”写成贡献。
- 使用 site/sequence split，绝不能随机切相邻帧。所有数据处理、split、配置和结果写入版本化
  文件。

## 论文级要求

最终需要补上可扩展的 exact-forward/path-cut 或 merge-tree 算子，并明确区分：

1. voxel marginal calibration；
2. joint occupancy calibration；
3. two-terminal, footprint-conditioned event calibration。

不要声称“没人做过”或“已经达到 ICRA/IROS”，除非完成系统文献核查和所有 baselines。每次
修改后运行测试，并在 `results/` 写清楚命令、硬件、随机种子、指标和失败条件。

## 交付格式

结束本轮时请给出：

1. 修改文件列表；
2. 完整运行命令；
3. 测试（含 skips）和 GPU smoke 输出；
4. P0 baseline 表格与图；
5. 是否通过 death test，以及继续/停止的理由。
