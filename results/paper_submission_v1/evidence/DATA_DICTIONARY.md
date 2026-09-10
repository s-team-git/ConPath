# Data dictionary / 数据字典

这个目录保存既有论文数字的来源、固定的开发验证分数，以及从既有缓存导出的**标量事件标签**。不含原始地图、模型权重、最终测试标签或新训练结果。所有比例使用 0–1；空值、空 CSV 单元和 `null` 表示缺失或不适用，不能读作零。本文中的“概率”是模型的可达事件分数，不是实机安全保证。

## 1. 数据集、重复与比较范围

| Cohort / 证据层 | 场景与实际 seed | 允许的解释 |
|---|---|---|
| D1 `parent_isolated_three_seed` | 100 train / 25 calibration / 40 validation 父地点；全来自物理 archive train；20260910、20260911、20260912 | 当前开发验证；训练最高 24 epoch，40 个验证地点后来参与开发判断，不能称最终测试或充分收敛 |
| D2 `coherent_two_seed_matched` | 同一地点划分，仅 20260910、20260911，所有比较方限制为这两个 seed | 相关分类噪声改动的探索性补充，不能与 D1 三种子平均混比 |
| H1 `historical_overlapping_parent_K128` | 160 训练 / 160 验证观测，142 个有查询的子场景，4224 事件；多数模型 20260831、20260901、20260902；direct-query 只有 20260831 | 27/160 验证**观测**与训练共享父地点，且历史曾读取物理 test。只保留聚合工程/机制记录；不能当新地点泛化 |
| M1 historical fixed marginal | H1 的固定 K128 采样世界，逐栅格独立打乱世界索引 | 实证边际完全相同的有限集合依赖干预；不是另一个独立训练模型，也不是干净新划分的因果训练结论 |
| S1 `synthetic_P0_template_holdout` | 既有 synthetic P0；full 为 20260827、20260828，no-event 为 20260827 | 单独的合成机制实验，场景、输入与 radius 协议不同，不能和 FlatLands 数字拼表 |

`seed` 是实际训练随机种子，不是地图样本编号。`repeats` 是该行实际重复数；规则控制的 `rule` 不是一次网络训练。`sample_sd` 是实际重复间样本标准差，不是置信区间、标准误或独立地点数量。单次重复的 SD 不填。

`K` 是归档预算标签，必须连同 `actual_map_count` 与 `K_role` 阅读。随机模型用 4 或 32 个真实二值世界；K4 是 K32 的固定前缀。确定性地图为 1 个世界。半径先验不生成地图，即使原文件叫 `predictions_k1.csv`，其 `actual_map_count=null`。条件均值阈值控制只产生 1 张二值图，`upstream_sampling_K` 表示上游平均用到多少样本。

## 2. 三类 CSV 的主键和字段

### 已保存分数 `sources/results/*/evaluation/*/predictions_k*.csv`

| 字段 | 定义 |
|---|---|
| `global_id` | 固定观测 ID；不要跨历史/新 cohort 仅按这个字段拼接 |
| `parent_group` | 数据源与父地点 ID 的组合，用于地点隔离与父地点加权 |
| `candidate_index` | 输入阶段冻结的候选查询编号，不是模型输出排序 |
| `radius_cells` | 机器人整数圆盘 footprint radius，单位为 **grid cell**；D1/D2 为 0、10、20，不转换为米 |
| `probability` | 固定模型/预算的原始事件分数；`>=0.8` 直接确定接受集合，不在验证集拟合新阈值 |

共 23 份，每份 1545 行，键为 `(global_id, candidate_index, radius_cells)`。每份覆盖同样 515 个查询及三个半径。原文件不含真值，打包时逐字节复制。

### 开发验证标签 `sources/results/paper_submission_v1/analysis/targets.csv`

主键同上；另含 `source_dataset` 与 `target`。`target=1` 表示既有 footprint-aware 精确几何评价判定存在路径，`0` 表示不存在。标签导出不重新执行几何或网络；仅从 snapshot 已注册、密封的 40 个 physical-train validation 缓存读取原 `candidate_indices` 与 `targets` 数组。`target_export_receipt.json` 绑定源包 SHA、允许的数组、源协议与导出脚本 SHA。这个包只携带标签 CSV，不携带 NPZ。它不含 25 个 calibration 地点的逐查询预测，也不能据此补齐缺失的 calibration NLL/ECE。

### 聚合和分层数据视图

- `sources/results/paper_validation_snapshot.csv`：29 个 cohort / method / K 聚合行。`*_mean`、`*_sample_sd` 对应冻结 JSON 的已有统计，`source_paths` 与 `source_sha256` 是原来源，不会被悄悄改写；包内位置见 `path_mapping.json`。
- `sources/results/paper_submission_v1/analysis/development_metrics_by_seed_and_stratum.csv`：按 seed 展开既有 snapshot，保留 `snapshot_json_pointer`、`source_prediction`、`source_prediction_sha256`、`validation_only`。`event_count` / `parent_count` 是该子集计数。
- `sources/results/paper_submission_v1/analysis/primary_three_seed_strata.csv`：原三种子主方法的既有分层值的平均/样本 SD；`*_available_repeats` 明确对应每个指标的非空重复数。
- `stratum_dimension` / `stratum_value` 指出 overall、来源、radius 或真值分组。`positive_rate` 是本分组按声明权重计算的正例比例，`mean_score` 是平均模型分数。
- `numeric_index.csv` 的 `snapshot_pointer` 是标准 JSON Pointer（`~0` 和 `~1` 分别转义 `~` 和 `/`）；`value_json` 为原始未舍入 JSON 数值；`section` 是顶层来源段。它只展开数字，没有重新计算性能。
- `row_catalog.json` 为 29 行提供方法、实际 seed、K、证据层、来源 JSON Pointer。新划分没有的 direct-query/no-event/no-global 列在 `missing_current_methods`，不会制造零分占位行。

## 3. 指标的精确定义

设事件标签 `y` 为 0/1，保存分数 `p` 位于 [0,1]，规范化权重为 `w`。

| 指标键 | 定义与注意事项 |
|---|---|
| `brier` / Event Brier | Σ w(p−y)²；越低越好 |
| `nll` / Event NLL | −Σ w[y log p + (1−y) log(1−p)]；仅计算 log 时 clip 到 [1e−6, 1−1e−6]，分数本身不修改 |
| `ece` / Event ECE | 十个等宽置信度桶，各桶权重 × 该桶平均置信度与平均标签之差的绝对值；p=1 放末桶；空桶无贡献 |
| `false_safe_at_0_8` | 在原分数 p≥0.8 的**已接受**事件中，标签为 0 的加权比例；未接受任何查询则为 `null` |
| `coverage_at_0_8` | 原分数 p≥0.8 的加权事件质量；允许为零 |
| `risk_at_30_percent` | 按原分数从高到低接受 30% 加权事件质量后的负例比例。临界并列分数组作为整体按相同比例接受，不用标签排序。不是 confidence=0.3，也不是 confidence=0.8 下的风险 |
| `radius_monotonicity` | 同一查询三个半径的原分数是否满足 p(0)≥p(10)≥p(20)，标签也检验相同次序。此结构性质不等价于事件校准好 |
| `observed_evidence_violation` | 已保存世界是否违背观测区域；这里保留既有审计报告，不新读取地图重跑几何 |
| `valid_support_violation` | 已保存世界是否在有效支持范围外产生开放栅格；这里同样仅保留原审计范围 |

`equal_parent` 权重为 1/(该分组有事件的父地点数 × 该父地点在该分组的事件数)，在每个分组内重新计算。`pooled` 给每个事件相同权重。`source_macro` 是另一个按来源做宏平均的已有汇总，不能和 pooled 或 equal-parent 混称“平均”。旧 H1 使用其历史子场景权重，不是独立父地点权重。

原 D1 登记的主指标为 Event Brier 与 risk30；NLL/ECE/@0.8 与分层汇总是既有固定分数的事后描述性分析。打包与离线核验不把它们改为预先注册的选模指标。保存的置信区间直接来自旧报告，没有重新 bootstrap。

## 4. 地图 Brier 必须分开

| 字段 | 被评估的地图量 |
|---|---|
| `sample_vote_cell_brier`、`hidden_map_binary_vote_brier` | K 个**实际二值世界**的 free-frequency，对 unknown AND valid-support 单元计算；确定性 K1 对应二值图误差 |
| `continuous_map_brier` | 单独保存的连续条件分类概率均值（随机模型为 32 次 latent draw），或确定性 sigmoid；不能改名为 K4 连续估计 |
| `all_cell_binary_vote_brier` | 某些合成来源的全栅格指标，不能与 hidden-only 混用 |
| `mean_iou` / `hidden_free_iou_mean` | 世界样本的 **free 类** IoU 均值，不是多类别 mIoU |
| `mean_blocked_iou` / `hidden_blocked_iou_mean` | blocked 类 IoU 均值，单独命名 |

当前地图原始数组未收入包，所以离线 verifier 只验证这些数字的来源、SHA 与精确索引；不声称重新从栅格计算了地图 Brier/IoU 或重跑了几何审计。

## 5. 核验范围和不可补齐部分

`python verify_offline.py` 只使用 Python 标准库和本目录文件，核验库存 SHA、物理 train 分区元数据、导出标签来源回执、23 份分数的完整键集合，以及两种权重下的已保存事件指标。它不联网、不导入模型代码、不读取目录外实验资产、不计算新置信区间。`sources/` 中 `.py` 和 snapshot 内嵌 builder 是原实现文档；核验器不会导入或执行它们。

历史报告可能仍保留后来被撤回的 `test_evaluated=false` 或 “clean” 字样；必须与 `flatlands_read_scope_erratum.json`、`flatlands_parent_group_audit.json` 和论文限制一起读。打包这些聚合报告及历史访问标识元数据不代表新读取历史 test 标签。外部模型未充分收敛，当前三种子主划分的 direct-query/no-event/no-global 仍缺失，calibration 逐查询预测仍缺失；本包没有补做这些实验。
