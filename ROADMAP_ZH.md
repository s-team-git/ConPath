# ConPath 整体架构与执行路线

这份文档回答三个问题：现在代码做了什么、下一步具体做什么、做到什么程度才值得继续写论文。

## 一句话目标

输入不完整的环境观测，学习一组空间相关的可能地图，并输出：

```text
给定起点 s、终点 g 和车辆 footprint 半径 r，至少存在一条可行路径的概率 q(s,g,r)。
```

论文不以 RGB-LiDAR 融合、A* 或机器人部署为创新点。核心是：

```text
联合随机地图后验 + footprint 条件连通事件 + task-level proper calibration
```

## 总体数据流

```mermaid
flowchart LR
    A[公开数据 RGB / LiDAR] --> B[数据集官方或标准 BEV Encoder]
    B --> C[BEV feature F]
    C --> D[随机场均值 logits mu]
    C --> E[低秩全局相关因子 B]
    C --> F[局部相关尺度 sigma]
    D --> G[K 个相关随机地图]
    E --> G
    F --> G
    G --> H[潜在世界 TRAVERSABLE / BLOCKED]
    H --> I[车辆圆盘 footprint 腐蚀]
    I --> J[四邻域 max-min 连通传播]
    J --> K[q(s,g,r) 与 sample events]
    L[完整 GT 地图] --> M[离线最大净空标签 C*]
    M --> N[多半径可达标签]
    N --> O[U-statistic Brier / CRPS]
    K --> O
    L --> P[posterior-marginal NLL 与 variogram]
    G --> P
```

## 当前代码的真实输入

P0 原型不是伪造一个尚未完成的相机投影系统，而是输入已经栅格化的 BEV 观测：

```text
observation_bev: [B, 3, H, W]              # 通用接口

channel 0 = 当前明确观测到的 traversable
channel 1 = 当前明确观测到的 obstacle
channel 2 = unknown / occluded
```

合成数据故意遮住决定全局连通性的门洞。开门与关门样本拥有相同可见输入，但完整 GT 拓扑不同。
正式 P0 已加入两个会改变门洞先验的可见 context family。为了与收到 context metadata 的基线
保持相同条件信息，P0 默认再广播一个 context-bit plane，输入因此为 `[B,4,H,W]`；该 plane 不含
门洞实现或事件标签。通用/真实数据 `forward_features(...)` 接口不依赖这一合成专用通道。

真实数据阶段则是：

```text
RGB + LiDAR -> 官方/标准感知 backbone -> BEV feature
                                         -> PathRelNet.forward_features(...)
```

感知 backbone 可替换，论文对所有方法固定同一 backbone。

## 网络输出

| 输出 | Tensor shape | 用途 |
|---|---|---|
| `mean_logits` | `[B,2,H,W]` | logistic-normal 的均值参数，不冒充后验边际 |
| `posterior_marginal_probs` | `[B,2,H,W]` | conditional softmax 跨 latent 样本的均值 |
| `factor_maps` | `[B,2,D,H,W]` | 长距离空间相关性 |
| `local_scale` | `[B,2,H,W]` | 局部随机场尺度 |
| `sample_probs` | `[B,K,2,H,W]` | K 个 straight-through 完整地图样本 |
| `sample_reachability` | `[B,K,Q,R]` | 每个地图、查询和车体半径的可达事件 |
| `reachability` | `[B,Q,R]` | 最终可达概率曲线 |

潜在世界类别顺序固定为：

```text
0 = TRAVERSABLE
1 = BLOCKED
```

只有类别 0 能进入安全配置空间。`UNKNOWN` 仍可以作为输入观测通道，但不是物理世界类别；没有可靠完整真值的格子使用 valid/ignore mask，不参与标签和 query 生成。

## 标签怎样生成

对每张完整 GT 地图：

1. 对完整有效 GT，将 blocked 和图外区域设为 occupied；无真值区域不生成查询。
2. 计算每个 free cell 能容纳的最大离散圆盘半径。
3. 对每个起终点运行 maximum-bottleneck graph search：

   ```text
   C*(s,g;M) = max_path min_cell clearance(cell)
   ```

4. 对每个 footprint 半径 `r` 生成：

   ```text
   y(s,g,r) = reachable AND [C* >= r]
   ```

`src/pathrel/labels.py` 已实现这个离线真值算法，并有阻断、对称性、半径单调性和端点非法测试。

## 如何训练

下面是正式实验计划。通用 `scripts/train_synthetic.py` 仍是缩短的代码路径；P0 专用
`scripts/train_p0_neural.py` 已实现 mean-map warm-up、重复世界分组监督和随后联合训练。公开数据
阶段仍需按数据规模明确 encoder 冻结/解冻调度。

### 阶段 A：平均地图 warm-up

训练：

```text
Tiny/official encoder + mean_logits
```

损失：

```text
L_warmup = CrossEntropy(mean_logits, GT classes)
```

目标是先让确定性地图稳定。

### 阶段 B：学习联合随机后验

开启：

```text
factor_head + scale_head + K map sampling
```

联合阶段不使用 `softmax(mean_logits)` 冒充后验边际，加入：

```text
L_posterior_marginal_NLL
L_variogram
L_reachability_U-statistic_Brier
```

普通有限样本均值的 Brier 会额外惩罚 Monte-Carlo 方差，因此训练代码实现了独立样本 U-statistic；论文评价仍报告标准 Brier、NLL 和 ECE。

### 阶段 C：小学习率联合微调

解冻 encoder 最后几层，固定 query sampling protocol，联合训练。训练建议 `K=8`；验证至少 `K=32`；最终结果检查 `K=32/64/128` 收敛性。

## 当前完成度

| 模块 | 状态 | 对应文件 |
|---|---|---|
| exact GT clearance/reachability | 已完成并运行测试 | `src/pathrel/labels.py` |
| 相关随机地图 decoder | 已完成 | `src/pathrel/stochastic_decoder.py` |
| footprint + max-min layer | 已完成原型 | `src/pathrel/reachability.py` |
| prototype scoring losses | 已完成 | `src/pathrel/losses.py` |
| 端到端 core model | 已完成 | `src/pathrel/model.py` |
| 合成歧义数据 | 已完成 | `src/pathrel/synthetic.py` |
| forward/backward smoke | 已完成 | `scripts/` 与 `tests/` |
| ORFD adapter | 未开始 | P1 |
| FlatLands completion/query audit | 512 场景 data gate、direct-ZIP adapter、统一 evaluator 与首轮 validation baseline 已通过；最终结果待多 seed/扩展性/第二域 | P1 |
| UnScenes3D encoder/loader | 未开始 | P2 |
| WildOcc cross-domain | 未开始 | P2 |
| scalable path-cut bounds | NumPy merge-tree 已有单图 exact reference，新增 batch×sample×query 封装与 CPU contract benchmark；共享起点传播和 ConPath public-data 入口已接通，CUDA/soft-backward 未完成 | P3 |
| SE(2) 矩形 footprint | 未开始 | 2.5-D 版本成立后再做 |

## 接下来按什么顺序做

### P0：把合成实验做成 death test

1. 增加至少两种可见 context family，使 `P(door open | context)` 明确不同，同时每种 context 重复采样多个隐藏世界。
2. 训练当前相关模型。
3. 增加 per-query constant、相同边际的 independent-Bernoulli 和 direct-query MLP baseline。
4. 报告 map IoU、voxel ECE、reachability Brier/ECE。
5. 检查随机样本是否生成“门洞整体开/整体关”，而非独立椒盐噪声。

`scripts/evaluate_p0.py` 使用两个可见 context family（隐藏门洞先验约为 0.2/0.8）、按 scene
template 留出测试集，并保存 JSON/CSV/SVG。脚本中的 `PathRel_correlated_event` 仍只是相关后验
oracle 代理；学习模型由 `scripts/train_p0_neural.py` 在同一 split/query protocol 下单独验证。

P0 已于 2026-08-30 通过：完整模型两个优化种子的 event Brier 为 `0.1164/0.1116`，均优于
independent `0.1832` 与 direct-query `0.1699`；匹配的 no-reach 对照为 `0.1914` 并失败。
两个完整种子也通过地图质量、ECE 和 context-gap 门槛。详见 `P0_DEATH_TEST.md` 与
`CONTINUATION.md`。该结论只允许进入 P1 数据审计；合成样本仍有约 13.7%-16.1% 门洞碎裂，且
不能视为公开数据或论文结果。

### P1：公开数据与事件可辨识性审计

1. 优先审计 FlatLands 的 partial/full/valid mask、natural query、断连、窄瓶颈和 footprint
   分布，并把其 stochastic/flow completion 作为强 baseline。
2. 再冻结 ORFD 数据版本与 sequence split，明确 traversable、obstacle、unreachable/unknown
   映射。
3. 自动采样只依赖观测和 goal 的 query，同时保留数据分布权重。
4. 统计 disconnected、窄瓶颈、多路径 query 占比；若有效负例或瓶颈少于约 10%-15%，该数据
   不能作为主事件 benchmark。
5. 固定一个 encoder，只比较 posterior/reliability 方法。

若有效负例或瓶颈 query 少于约 10%-15%，ORFD 只作辅助，不作为主数据。

2026-08-30 的 FlatLands bounded audit 已完成：官方 observation split 因 scene leakage 继续
NO-GO；非官方 `provenance.original_split` 在 512 个不同场景、16 个 split/source strata 上通过
mask 与最低 query-balance gate。4,653 个有效端点中有 121 个 radius-0 断连、3,095 个足迹失败和
1,437 个 20 cm 正例。该分布并不均匀：test/ARKitScenes 的 115 个有效 query 在 20 cm 下没有
正例。direct-from-ZIP loader 已完成 512/512 场景回放、双进程加载与防 split 泄漏测试，审计
JSON/图表也已同步到项目网站。首轮 validation-only 对照已完成：radius prior 的 Brier 为
0.15870，deterministic completion 为 0.08556，independent-cell K=32 为 0.22546，direct-query
为 0.09119；这些数字不是最终论文结果，test 仍锁定。下一步是多 seed ConPath、逐项消融、可扩展
connectivity、calibration/false-safe 与第二数据域，仍必须按 source/radius 报告。

### P2：正式公开数据实验

若 FlatLands 固定基线证明该事件任务可学习，先在明确标注为非官方的 provenance split 上测
“completion samples 到路径事件”的校准；不得使用存在 scene leakage 的公开目录 split。
若室内 floor-map 不足以代表小车支撑面，再使用 UnScenes3D 的 occupancy 与 road elevation
构造 2.5-D 地图，WildOcc 做跨域测试。必须按场景/矿区/序列拆分，禁止相邻帧随机拆分。

### P3：论文完整算法

当前迭代传播在 `H*W` 次时精确，固定较少步数只是有界近似。`labels.py` 已增加 exact-forward
的 NumPy merge-tree 参考实现，并用随机阈值连通性测试验证；现在又提供
`batched_merge_tree_bottleneck_scores([B,K,H,W], [B,Q,2], [B,Q,2])`，避免每个 query 重复构建
图。64×64、4 个 sample 的 CPU contract benchmark 在 32/128/512/2048 query 下分别约为
0.036/0.037/0.044/0.060 秒，输出与单图 oracle 逐项一致；这些是合成效率诊断，不是论文结果。
它还不是可反传 CUDA 算子。最终应研究：

- merge-tree / maximum-spanning-tree 的批量查询；或
- top-K path 与 cut 的上下界；或
- exact-forward、soft-backward 的高效自定义算子。

这一步才是最终算法深度的主要来源之一。

## 最终实验必须回答的四个问题

1. 相同 voxel mIoU/ECE 下，联合模型是否显著降低 reachability Brier/ECE？
2. 高置信度预测中的 false-safe rate 是否下降？
3. 优势是否集中出现在真实遮挡、瓶颈和替代路径场景，而不是人为 query？
4. 更换车辆半径和测试地点后，概率是否仍然校准？

如果这四个问题没有正面答案，就不应继续扩展 RGB backbone、3DGS 或实车系统。

## 从现在到论文定稿的完整任务链

以下顺序是当前持续 goal；每一阶段必须留下配置、随机种子、机器可读报告、图表和网站快照，
不能因为后面的结果更好而修改前面的 split 或 query。

1. **P1 评测合同与强基线（当前已完成首轮）。** 冻结统一 evaluator、scene-weighted 指标、
   bootstrap 单位和 train-only 拟合边界；已完成 deterministic completion、independent-cell
   posterior、direct-query predictor 与 train-only radius prior。首轮结果仅用于诊断，下一轮补
   多 seed、K 收敛和强官方 completion baseline；继续按 split/source/radius 输出 Brier、NLL、ECE、
   false-safe 与效率。
2. **可扩展连通算子。** 将已验证的 merge-tree exact-forward 参考推进为批量实现，或采用
   exact-forward/soft-backward 算子；证明与离线 oracle 一致，报告显存、时间和 query 数扩展。
3. **ConPath 正式训练与消融。** 至少三个固定种子；比较完整模型、无 event loss、无全局相关
   因子、independent decoder、不同 K 和确定性均值图。所有方法共享 encoder、数据与 query。
4. **校准与安全分析。** 做 source/radius reliability、false-safe 阈值曲线、K 收敛、scene
   bootstrap 置信区间、失败案例和饱和 stratum 报告；不得用 pooled 均值掩盖 ARKitScenes。
5. **外部有效性。** 根据 P1 结果选择一个第二数据域；优先补足真实支撑/越野语义，而非再做一套
   相似室内数据。冻结 scene/site/sequence split，并只复用已经定稿的方法与超参数规则。
6. **可复现冻结。** 固定环境、配置、checkpoint、数据哈希、运行命令和最终表格；从干净环境
   重跑关键结果，网站同步所有可公开 JSON/CSV/SVG 与定性图片，并逐项核对 license/attribution。
7. **论文写作与投稿。** 先冻结研究问题、贡献与 claim boundary，再完成方法、实验、相关工作、
   局限和附录；制作主表/主图，完成内部反例审稿、统计审计、匿名化与 ICRA/IROS 格式检查。

进入论文定稿的硬门槛是：强基线完整、完整模型多种子优势稳定、false-safe 有实质下降、关键消融
支持因果解释、第二域不反转主要结论、所有数字可从冻结报告自动生成。未达到这些门槛时，只能写
working draft，不能宣称已经具备投稿级证据。
