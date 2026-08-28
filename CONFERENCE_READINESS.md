# ConPath 顶会投稿可行性与门槛

## 当前判断

当前仓库是 synthetic contract prototype，不是可直接投稿的 ICRA/IROS 论文。它已经有一个
有潜力的科学问题，但尚未有公开数据结果、独立基线、任务级校准结果或可扩展的大图算法。

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

若 P0 中 calibrated independent-cell、direct-query 或 FlatLands completion + post-hoc
connectivity 追平 PathRel，停止该方向；若 P0 通过，再投入 UnScenes3D/WildOcc 和大规模算子。
不要在 death test 之前加入 3DGS、
ROS、实车闭环或更多传感器。

## 本轮 P0 审计状态（2026-08-27）

已实现 `scripts/evaluate_p0.py`，并按 scene-template 留出测试集、两个可见 context family
（隐藏门洞先验约 0.2/0.8）、多隐藏世界重复、常数/独立 cell/direct-query/edge-connectivity/
random completion/deterministic/correlated ablation 等基线。默认测试集结果为：direct-query
Brier 0.1692、deterministic threshold 0.1181、相关事件代理 0.0987；相关代理的 ECE 为
0.0373，独立 cell 为 0.1954，且地图边际 Brier 与独立采样相差 0.0173。因而 **oracle
proxy death test PASS**，支持继续验证联合后验假设；但 CUDA 上的神经 PathRel 尚未运行，当前
项目仍保持 **NO-GO**，不能接入公开数据或宣称 ICRA/IROS 贡献。

同时加入 `labels.py::merge_tree_bottleneck_scores` exact-forward NumPy 参考，用于后续可扩展
CUDA 算子的契约验证；这不是已经完成的可反传大图实现。
