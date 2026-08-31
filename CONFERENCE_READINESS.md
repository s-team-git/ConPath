# ConPath 顶会投稿可行性与门槛

## 当前判断

当前仓库仍是 synthetic contract prototype，不是可直接投稿的 ICRA/IROS 论文。神经 P0 已在
两个优化种子上通过，并有匹配的 no-reach 对照；但尚未有公开数据上的任务级校准结果、完整
独立强基线或可扩展的大图算法。

## 时间选择（截至 2026-08-28）

- ICRA 2027 的官方 paper deadline 是 2026-09-15 23:59 Pacific；以当前原型状态不应为了赶
  这个 deadline 拼凑结果。官方页面：
  <https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/>。
- IROS 2027 的官方 paper deadline 是 2027-03-01；它更适合作为第一目标。
  官方页面：
  <https://www.ieee-ras.org/event/2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-70525/>。
- 如果 IROS 结果显示问题成立，再准备 ICRA 2028 或 RA-L/会议转投版本。

## 论文必须证明什么

1. 在相同 voxel marginal 质量/ECE 下，联合后验显著降低 start-goal-footprint 事件的 Brier、
   NLL、ECE 和 false-safe rate。
2. 优势来自空间联合结构，而不是更大的 encoder、更多 Monte-Carlo 样本或阈值调参；必须
   包含 independent Bernoulli、deep ensemble、direct-query、deterministic threshold 和
   topology-loss 基线。
3. 在遮挡、窄瓶颈、断连和多替代路径上成立，并跨 sequence/site、地图和车辆 footprint 泛化。
4. 把当前 Python 迭代原型替换成可扩展的 exact-forward/merge-tree/path-cut 算子，或给出
   清晰的上下界；否则真实分辨率实验会被速度/显存问题卡住。
5. 真实数据只使用可审计的 occupancy/road-support 标签，并明确 unknown 与不可见区域，不能
   把合成门洞实验冒充真实导航验证。

## 主要审稿风险

相关 occupancy 建模、拓扑规划、edge/connectivity learning 和 occupancy 不确定性规划本身已有
先例。因此论文不能把“RGB/LiDAR + 随机地图 + A*”作为贡献；贡献必须集中在联合后验到
`P(存在 footprint 条件路径)` 的事件级 proper calibration，以及它相对于 voxel calibration
的可验证差异。

至少要在 related work 中正面讨论 [MRFMap](https://www.roboticsproceedings.org/rss16/p060.html)、
[Saroya et al.](https://doi.org/10.1109/LRA.2021.3068886)、
[Banfi et al.](https://arxiv.org/abs/2205.14251) 和
[Ma et al.](https://arxiv.org/abs/2112.08106)，并把 connectivity-loss/edge predictor 纳入
baseline；还要检查 [SCOPE](https://arxiv.org/abs/2407.00144) 与
[diffusion-based occupancy completion](https://arxiv.org/abs/2409.10681)。尤其要把
[FlatLands](https://arxiv.org/abs/2603.16016) 作为强近邻和优先数据审计对象：它已经提供
partial-view BEV、多个合法完整布局和 stochastic/flow completion benchmark。不能声称首次
提出 correlated occupancy、topology-aware planning、connectivity learning、partial-view
multi-layout completion 或 stochastic map completion。

FlatLands 也必须先做 query-balance audit：若自然 reachable/unreachable、窄瓶颈或替代路径
太少，它只能作为 completion/OOD 基线，不能直接支撑导航事件结论。

## Go / No-Go

若 P0 中 calibrated independent-cell 或 direct-query 追平 PathRel，停止该方向；当前学习版
P0 已通过，所以只允许进入 **P1 数据与事件可辨识性审计**。P1 若发现 FlatLands completion +
post-hoc connectivity 追平，或自然断连/窄瓶颈 query 不足，仍应停止或更换主数据，而不是直接
投入 UnScenes3D/WildOcc 和大规模算子。此阶段不要加入 3DGS、ROS、实车闭环或更多传感器。

## 本轮 P0 审计状态（更新于 2026-08-30）

已实现 `scripts/evaluate_p0.py`，并按 scene-template 留出测试集、两个可见 context family
（隐藏门洞先验约 0.2/0.8）、多隐藏世界重复、常数/独立 cell/direct-query/edge-connectivity/
random completion/deterministic/correlated ablation 等基线。最新默认测试集结果为：direct-query
Brier 0.1699、deterministic threshold 0.1458、相关事件代理 0.1024；相关代理的 ECE 为
0.0325，独立 cell 为 0.1762，且地图边际 Brier 与独立采样相差 0.0173。因而 **oracle
proxy death test PASS**，支持继续验证联合后验假设。早期 120-step CUDA 神经 checkpoint 的
event Brier 为 0.2436，确实失败；随后修正了 scaled-Gumbel 边际、事件梯度、重复世界监督、全局
上下文编码和可见 context 输入，并加入可恢复检查点与严格 context-gap gate。

在完全相同的 12/4 template、24 worlds/template、128 validation-sample protocol 下，完整模型的
两个优化种子均通过：

| 配置 | Event Brier | ECE | Hard-map Brier | radius-0 context-gap ratio | 结论 |
|---|---:|---:|---:|---:|---|
| full, seed 20260827 | 0.1164 | 0.0786 | 0.00338 | 0.5735 | PASS |
| full, seed 20260828 | 0.1116 | 0.0719 | 0.00289 | 0.6957 | PASS |
| no-reach, seed 20260827 | 0.1914 | 0.1936 | 0.00310 | 0.2383 | FAIL |

两个完整种子都优于 independent (`0.1832`) 与 direct-query (`0.1699`) 的 event Brier；而
no-reach 对照在地图 Brier 仍好的情况下事件指标和上下文条件性同时失败。因此当前决策升级为
**P0 GO / P1 audit allowed**。这仍不是公开数据或论文级 GO：它只有一个固定 synthetic split、
两个优化种子，完整模型仍有约 13.7%-16.1% 的门洞碎裂。下一步必须先冻结并审计 P1 数据版本、
mask/标签语义、natural-query 分布和官方 completion baseline；在该审计通过前不得宣称
ICRA/IROS 贡献。

同时加入 `labels.py::merge_tree_bottleneck_scores` exact-forward NumPy 参考，用于后续可扩展
CUDA 算子的契约验证；这不是已经完成的可反传大图实现。
