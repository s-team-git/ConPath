# 现有证据的论文图片索引

外部训练已取消。本批仅重绘已有数值与保存世界，没有启动模型推理、训练或读取测试资产；不把外部短训练图放进论文主图。

不可变快照：[20260910T063236.828092Z](results/paper_figures/20260910T063236.828092Z/manifest.json)。每幅图均有SVG、PNG、PDF；完整源文件SHA256、协议、种子与半径见manifest。

主数据为100/25/40父级地点开发队列（40验证地点、515端点查询、1545半径事件）。原ConPath/independent/deterministic三个种子与coherent两个种子分组展示；验证反馈已用于开发，不能写成最终测试或外部方法优越性。

## 已生成图片

### 01_architecture

[PNG](results/paper_figures/20260910T063236.828092Z/01_architecture.png) · [SVG](results/paper_figures/20260910T063236.828092Z/01_architecture.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/01_architecture.pdf)

最终原版ConPath的实际代码顺序：先按已知类别/支持范围设logit±10，再加入尺度1的截断独立Gumbel并采样；不存在原硬地图采样后的通用投影步骤。

English caption: Implemented original ConPath pipeline. Observed and support-invalid cells are conditioned through logits of +10/-10 before unit-scale clipped independent Gumbel sampling. Correlation arises from Gaussian global factors and local logit fields. Exact hard-event correction uses a bounded relaxed backward surrogate; coherent categorical noise is a separately trained supplementary variant.

### 03_event_metrics

[PNG](results/paper_figures/20260910T063236.828092Z/03_event_metrics.png) · [SVG](results/paper_figures/20260910T063236.828092Z/03_event_metrics.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/03_event_metrics.pdf)

父级地点等权的事件指标，分开呈现两种子coherent比较和三种子原采样器比较。

English caption: Event metrics on the same 40-parent development cohort. Two-seed coherent comparisons and three-seed original-sampler comparisons are shown separately; error bars are training-seed SD, not confidence intervals.

### 04_reliability

[PNG](results/paper_figures/20260910T063236.828092Z/04_reliability.png) · [SVG](results/paper_figures/20260910T063236.828092Z/04_reliability.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/04_reliability.pdf)

原ConPath及两个基线三个匹配种子的原始事件可靠性；分箱内按父级地点等权计算。

English caption: Uncalibrated event reliability for the original ConPath and baselines, for each of the three matched training seeds, using ten fixed bins and equal-parent weights. Empty bins are omitted; line segments are visual guides.

### 05_risk_coverage

[PNG](results/paper_figures/20260910T063236.828092Z/05_risk_coverage.png) · [SVG](results/paper_figures/20260910T063236.828092Z/05_risk_coverage.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/05_risk_coverage.pdf)

相同覆盖率下的误判风险；不把较低覆盖率下的风险当作直接安全优势。

English caption: Selective risk versus weighted coverage, with label-independent fractional acceptance of tied boundary scores. Thin curves show individual seeds and thick curves their means; no deployment safety guarantee is implied.

### 06_radius

[PNG](results/paper_figures/20260910T063236.828092Z/06_radius.png) · [SVG](results/paper_figures/20260910T063236.828092Z/06_radius.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/06_radius.pdf)

固定查询在0、10、20格圆盘足迹下的事件误差。

English caption: Event Brier at the three frozen disk-footprint radii, in raster cells. Equal-parent weights are recomputed within each radius; error bars denote seed SD.

### 07_map_metrics

[PNG](results/paper_figures/20260910T063236.828092Z/07_map_metrics.png) · [SVG](results/paper_figures/20260910T063236.828092Z/07_map_metrics.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/07_map_metrics.pdf)

同一隐藏有效区域的地图指标；与路径事件评分区分。

English caption: Hidden valid-cell map metrics on the same cohort, separated from path-event scores. The empirical cellwise free frequency is not a path probability.

### 02_fixed_case_00

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_00.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_00.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_00.pdf)

固定哈希案例1，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 1, obs_014028:q024, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_01

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_01.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_01.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_01.pdf)

固定哈希案例2，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 2, obs_013470:q012, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_02

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_02.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_02.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_02.pdf)

固定哈希案例3，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 3, obs_082481:q030, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_03

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_03.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_03.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_03.pdf)

固定哈希案例4，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 4, obs_065777:q002, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_04

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_04.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_04.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_04.pdf)

固定哈希案例5，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 5, obs_114561:q002, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_05

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_05.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_05.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_05.pdf)

固定哈希案例6，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 6, obs_101106:q011, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_06

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_06.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_06.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_06.pdf)

固定哈希案例7，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 7, obs_166408:q019, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_07

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_07.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_07.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_07.pdf)

固定哈希案例8，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 8, obs_157856:q019, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_08

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_08.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_08.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_08.pdf)

固定哈希案例9，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 9, obs_203894:q002, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 02_fixed_case_09

[PNG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_09.png) · [SVG](results/paper_figures/20260910T063236.828092Z/02_fixed_case_09.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/02_fixed_case_09.pdf)

固定哈希案例10，同一输入、参考及同一种子的前三方法真实世界。

English caption: Fixed pre-existing gallery case 10, obs_173909:q007, radius 10 cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.

### 08_synthetic_event_ablation

[PNG](results/paper_figures/20260910T063236.828092Z/08_synthetic_event_ablation.png) · [SVG](results/paper_figures/20260910T063236.828092Z/08_synthetic_event_ablation.svg) · [PDF](results/paper_figures/20260910T063236.828092Z/08_synthetic_event_ablation.pdf)

P0已保存报告中的事件指标；两种子完整模型与一个匹配种子的无事件损失控制分别标识，不作跨数据集并表。

English caption: Existing synthetic held-out-template diagnostics, reported separately from FlatLands. The no-event-loss control has one matched seed; the additional full-model seed is shown separately, not treated as an extra ablation repeat.

## 不可生成或不采用的部分

- **父级隔离no-event/no-global/direct定性对照：不可生成。** 该cohort没有这些方法的已保存训练/世界结果；旧K128属于含地点重叠及历史physical-test访问的不同队列，不能补入。
- **P0 obs/reference/模型多世界定性：不可生成。** 三个P0目录仅有报告、日志和模型checkpoint，未发现已保存地图数组；本任务禁止新模型推理。
- **历史K128两例作为无选择偏差的消融主图：不采用。** 旧例子名称与既有选择记录含positive_recovery/false_safe_avoided，按结果挑选；不能把沿用旧挑图称为盲选。
- **严格窄瓶颈定性：不可生成独立瓶颈案例。** 只在既有10个固定查询中核对reference r0可达、当前半径不可达且起终点均放得下圆盘；不另按模型结果选图。

不可达案例已保留在原10张固定图中，均标参考事件；绿色连通不等于半径圆盘能通过。没有为凑正例/负例而替换固定查询。

架构图为代码路径示意，不含新实验成绩。原版与coherent均重新训练，independent同时移除全局因素并缩小局部核；这些比较不能作为同边际纯相关性的因果证明。

图中NLL/ECE是对已有CSV的事后描述性汇总；既有Brier和30%风险已逐run核对重现。没有在验证集拟合校准映射，也没有改变选中checkpoint。

## 唯一新增规则控制补图

[PNG](results/paper_figures/rule_controls/20260910T064132.442897Z/09_rule_controls.png) · [SVG](results/paper_figures/rule_controls/20260910T064132.442897Z/09_rule_controls.svg) · [PDF](results/paper_figures/rule_controls/20260910T064132.442897Z/09_rule_controls.pdf) · [来源与全部数值](results/paper_figures/rule_controls/20260910T064132.442897Z/manifest.json)

原主图03/04只比较训练模型；此图完整纳入已有四个规则控制，不按指标删掉不利结果。训练半径先验的NLL/ECE更低但0.8覆盖率为0；确定性模型的连续隐藏地图Brier为0.138685，低于ConPath的0.151468。风险与覆盖率需一起看。规则没有三次训练，不能制造三种子误差条。

English caption: All available deterministic rule controls and three trained methods on the same 40-parent development cohort. All rule methods are shown without outcome-based selection. Error bars for trained methods denote the sample SD of three seeds; rules have one deterministic evaluation. Undefined risk at zero accepted coverage is omitted. The training-radius prior has lower NLL and ECE but zero coverage at score 0.8. Deterministic completion has lower continuous hidden-map Brier (0.138685) than original ConPath (0.151468). No method dominates all objectives; development validation only.

## 历史RQ1 / RQ2聚合图（单独附录范围）

仅原字节复制已有SVG/PDF；不读取旧世界、目标、原始图片，不重算或重新选样。**旧文件名与图标题中的clean不代表地点完全隔离：160个旧验证观测中有27个与训练观测共享父地点，且存在历史physical-test归档访问。** 142是贡献事件的历史子场景数，不是已证明独立的父地点数。原test-untouched标记已撤回；这些图不能用作新地点泛化主图或外部正式比较。

- RQ1 固定边际重排：[SVG](results/paper_figures/historical_aggregate/flatlands_clean_marginal_shuffle.svg) · [PDF](results/paper_figures/historical_aggregate/flatlands_clean_marginal_shuffle.pdf)。English caption: Historical exact-empirical-marginal shuffle diagnostics at K=128. Evidence concerns spatial arrangement conditional on this mixed development cohort; parent overlap and prior physical-test archive access preclude a new-location generalization claim.
- RQ2 训练消融：[SVG](results/paper_figures/historical_aggregate/flatlands_clean_training_ablations.svg) · [PDF](results/paper_figures/historical_aggregate/flatlands_clean_training_ablations.pdf)。English caption: Historical three-seed training ablations on the same mixed development cohort. Removing event loss retains the common event-based checkpoint-selection rule; removing global decoder factors retains encoder global context. These are not current parent-isolated qualitative or final-test results.

[历史文件逐字节SHA清单（计数单位更正）](results/paper_figures/historical_aggregate/manifest_v2.json) · [历史测试访问更正](site/data/flatlands_read_scope_erratum.json)。原按结果挑选的旧定性图仍不采用。

本轮真实目视复核：[18张PNG逐图记录](results/paper_figures/visual_reviews/20260910T064523.389507Z/review.json)。54个当前图导出SHA通过，历史4个SVG/PDF只做原字节核验；不把版面检查当成模型质量或正式优越性结论。
