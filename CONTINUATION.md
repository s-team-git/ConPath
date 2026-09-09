# ConPath continuation state

## 当前交付：模型效果首页重排（2026-09-09）

用户反馈效果图难找、网站混乱，并询问下一步。本轮只完成网站信息结构调整与建议记录：
`index.html`提供三栏输入/预测/参考、两个固定旧案例、实际补全/概率切换、四行概览和下一步；
`research.html`保留原全部图表、数据浏览、消融、外部工程与更正。旧深链可跳转，研究页顶部可返回首页。
`build_model_home.py`从已发布SVG复制12张图，只更换文字标题，嵌入位图和起终点几何逐项相同；
没有重新解码原始数据用于渲染、重跑推理或新训练；资产审计会读取既有训练素材字节核对哈希。模型示例仍是原K=128解释性案例；新K=4统计单独说明。
首页与研究页桌面/手机真实浏览器通过；凭据见 `results/model_home_publication_20260909/completion.json`。
恢复时若发布收据已通过，不重做网站或旧评估。科研下一项仍先审查测试访问与父级划分，然后做干净小规模基线验证；
不能把本轮前端完成当成数据问题已解决。以下历史更正仍有效。

**读取范围更正：旧训练／验证包含原发布test中的8／5条观测，本轮回放也读取了这5条验证观测。更早的512观测查询审计检查过53条物理test观测（含32条ScanNet++）。因此撤回“物理test从未读取”的说法；旧 `test_evaluated=false` 等标志不能证明物理测试未触碰。详见[读取范围更正记录](site/data/flatlands_read_scope_erratum.json)。已停止进一步读取物理test图片，最终留出集需要审核全部历史访问并排除已检查的父级地点。**

This file is the durable hand-off for interrupted Codex sessions. Update it after every material
diagnostic, code change, or experiment; do not rely on chat history or ignored `results/` alone.

## 最新续接：2026-09-09

**2026-09-09 更正：旧 FlatLands 队列仅隔离子场景 ID，27/160个验证观测与训练共享建筑／地点。旧数值保留为队列诊断，不能证明新建筑泛化；旧子场景配对区间也不是独立建筑总体区间。数据隔离门槛重新打开。当前统一K=4对照、90条论文公开成绩、分组修正和历史帧分析见 [最新中文评估](BASELINE_REVIEW_ZH.md)。**

用户授权开始分析和改进，同时要求优先论文公开成绩、避免不必要复现。本轮已完成：

- `current_baseline_k4_v1`：12组现有模型/无训练规则，4,800张地图、50,688条事件预测；9份旧CSV回放零差异。K4 ConPath Brier0.08007，全可通行0.07237，风险30%为4.89%/13.43%；不能宣布全面胜出。
- `baseline_review_20260909_v1`：90条论文成绩、旧子场景配对统计、数值独立检查。论文MES不是通路Brier，所有原论文数字reference-only。
- `flatlands_parent_groups_v2`：官方父级分组确认Matterport19、3RScan5、ZInD2、ScanNet1、ARKit0条旧验证重叠；53条未知visit隔离。新候选215,289条、4,479组，仅元数据，不是正式训练准备完成。v1过严地下楼层规则已被v2取代。
- `unscenes_history_feasibility_v1`：9训练场景36帧27历史对；观测冲突22.75%，位姿候选提高平均重合但仅11/27中位距离改善，独立ego-pose仅4帧可核。没有未来输入、模型训练或测试图访问。
- 本輪论文、工作计划、恢复状态与网页同步更正；发布凭据在 `results/baseline_review_publication_20260909/completion.json`。存在通过收据时不要重复本轮回放或发布。

**下一项**：核查父级候选的训练侧输入/目标质量和盲查询，确认物理尺度、跨来源重复与独立最终测试设计；之后冻结共同正式规模，按公开成绩→作者预测/权重→必要同协议适配的顺序补强基线。室外先审计软观测与2.5D地面表达、历史位姿质量并做无训练因果融合，再启动预设的2×2注意力实验。不沿用旧检查点作新分组主表，不追加旧错误硬约束下的长训练。所有封存测试继续锁定；没有runtime goal或当前训练进程。

## 历史 Recovery snapshot（以下旧“下一项”受上述状态替代）



**最新续接（2026-09-08）：LaMa与FM+XAttn接入、短训练和有效批量64实测均完成。**
本轮响应“下一步”执行新模块工程阶段；此前CogniPlan与六组消融已完成，不重复运行。
`FLATLANDS_EXTERNAL_PROGRESS_ZH.md`记录固定源码、损失/骨干适配、生成次数、原始耗时与局限。
LaMa保留固定MapEx子模块Big-LaMa的18个FFC块（50,966,465生成器参数），FM+XAttn为文献重实现（21,202,753参数）。
两者共用`已观测可通行/未知/有效范围`输入，边界与证据投影一致；均为随机初始化短训练，无新预训练权重。
固定32个训练观测，第一套batch2为10次预热+100次记录，保存LaMa32张/Flow128张真实输出；
第二套实际有效batch64为5次预热+100次完整优化更新，LaMa微批16×4、Flow微批4×16。
完整更新平均1.353677/3.422191秒，显存11.81/5.31GiB；原版30万步仅训练外推112.81/285.18小时每模型。
这不是正式配方或已启动任务；其他GPU负载、全量数据I/O、验证和存档影响实际排期。
LaMa微批BatchNorm不能声称等同全批64/SyncBN；四成员集成尚未训练。
Flow的25步Heun、K4、CFG实际50次批量前向/400次单图等价前向，不沿用“步数×样本数”的漏计。
新进程回放连续/二值结果完全一致，321项审计与111项测试通过；已查看全部32例，隐藏区域仍有明显错误。
训练损失下降和短训练地图仅作接口诊断，不作为充分收敛的外部基线或正式比较分数。
网站新增固定名单前两例共16张中文标注PNG、两组SVG/PDF训练曲线；图片batch2和曲线batch64来自不同检查点，均披露。
桌面1440/手机390浏览器检查通过，修正新增表格手机横向溢出，保留图片放大和逐图图例。
原始记录：`results/flatlands_{lama,flow}_{profile,batch64}_v1/`；发布/回放/图像/浏览器：
`results/flatlands_external_publication_20260909/`（浏览器最终为`browser_v2/report.json`）。
所有短训练与回放已退出，没有正式长训练；不要重复这些profiles。三种外部方法接口现在都已接通。
**下一项：冻结共同正式数据规模与查询，预先约定阶段预算和留出收敛检查；补齐正式训练中断恢复，之后执行三次独立重复。**
扩大数据须同时重训ConPath和强简单对照；不把集成成员当独立重复，不能用欠收敛对手支持论文优势。
CogniPlan母地图2400/300/300划分不变，FlatLands物理尺度仍未核实，半径保留格单位。
FlatLands物理test、UnScenes location6及CogniPlan测试资产仍锁定，没有注册runtime goal。
本轮Git/网站是否完成以`results/flatlands_external_publication_20260909/completion.json`为准；
若收据存在且已核验，下一轮直接推进共同数据和收敛阶段，不重做发布。

以下为此前已完成阶段的历史记录：

**最新续接（2026-09-08）：已响应“开始下一步”，外部比较的CogniPlan工程阶段完成。**
六组消融已发布并线上核验，提交37de858；不要重复旧发布或重跑旧矩阵。
新代码 `src/pathrel/cogniplan.py` 复用 `third_party/cogniplan/` 固定官方源码，保留MIT许可证；
训练更新与官方循环逐参数回放通过。32个固定训练观测生成128张图，原生后处理精确一致，观测零冲突。
新母地图/D4组划分2,400训练/300校准/300验证，观测数19,045/2,369/2,381。
重建/对抗阶段各10次预热+100次测量均完成，短训练已退出；原版50万步仅训练成本外推12.45小时/seed、
三次串行37.34小时，不是正式已启动任务或论文整体ETA。当前GPU存在其他项目负载。
v1原生cuDNN自动算法跨进程有33个二值单元变化；v2关闭自动测速/TF32、开启确定性算法并完整重测。
原6小时估算已由v2约12.45小时替代，历史收据保留；禁止混用执行设置成本。
本轮PyTorch实际为2.12.1+cu130；以profile环境记录为准，不沿用历史环境版本文字。
报告 `EXTERNAL_PROGRESS_ZH.md`；原始产物 `results/external_protocol_v1/`、`results/cogniplan_native_profile_v2/`，
发布/审计 `results/external_publication_20260908/`。103项测试通过，全部32例视觉检查保留较差预测。
网站新增3类固定首例、21张中文标注图，明确公开权重原训练数据接口检查不属正式对比。
FlatLands160份训练元数据均与裁剪一致，但物理标定尚无独立证据，保持格单位；
FM+XAttn损失已在补充S5确认为masked MSE，LaMa损失与子模块固定版本见新报告。
下一项：LaMa原生FFT骨干/条件流匹配接入，冻结CogniPlan共同查询与评价器，实际成本/收敛检查后冻结正式配方并启动三次训练。
本轮未启动正式长训练；所有最终测试仍锁定。以本轮发布完成收据判断Git/网站是否已同步。

以下为已完成的上一轮记录：

最新工作（2026-09-08）：用户要求完成评估汇总并继续收尾。本轮六组消融已于
08:56:19 UTC（04:56 EDT）完成精确评估、冻结审计与配对统计；调度器3081128和全部worker已退出。
不要重复启动这套已完成矩阵。当前任务是完成中文汇总、论文、网站与Git发布，然后进入已确认的外部实验接口/尺度/测速阶段。

三种模型的三次训练 Brier：完整0.06749±0.00936、no_event 0.20425±0.00322、
no_global 0.09499±0.00157。六个逐种子Brier区间均为正，支持完整模型；
六个等30%覆盖率风险区间均包含零，不支持稳定安全收益。no_global的风险点估计在两个种子反向。
no_event停在12/9/17轮，选择4/1/9；no_global停在12/18/18轮，选择4/10/10。
半径20格的no_event全部预测为0，但场景等权有路率20.67%；该失败诊断不能扩大尚未核对的物理尺度主张。

权威统计：`results/paper_clean_ablation_matrix_v1/analysis/report.json`。
中文报告：`EVALUATION_SUMMARY_ZH.md`；论文新增5.5节；网站独立三模型表和中文区间图。
两个此前固定的验证例子各三模型，输出6幅概率图和6幅首个真实采样按机器人尺寸收缩后的图，未按新结果挑例或挑样本。
图片采用CPU新绘图随机流；标题事件概率来自原CUDA精确验证CSV，随机流不同，不能要求两者精确相等。
有效绘图记录：`results/ablation_publication_20260908/cases_v2/report.json`（初版概率图记录在cases/，保留但不作当前来源）。
像素12/12、首个世界连通性6/6、9个预测CSV主指标/分层/配对Brier重算通过；98项测试通过。
最后桌面1440/手机390交互检查通过：`results/ablation_publication_20260908/browser_final/report.json`。
本轮独立数值/网页/版本/部署收据统一在 `results/ablation_publication_20260908/`；以完成收据判断是否已推送与线上核验。

已确认实验方案长期有效，见 `EXPERIMENT_DESIGN_ZH.md` 第8节；只将当前有界验证消融标记完成。
必做主比较是FlatLands原生LaMa/4成员集成、FM+XAttn和CogniPlan原生地图生成模块，
并保留强简单对照、三次独立重复、成本曲线、场景统计和失败分析。外部方法尚未运行。
先做数据/接口、原生质量、固定训练样本测速与具体配置冻结；扩大数据时全部相关方法另开一致的新版本。
正式测试仍锁定；不能因续训授权跳过这些科学门槛。没有注册runtime goal。

Current website: Chinese, minimal academic layout based on the user's Lightweight-3DGS
reference. `site/index.html` uses `styles-zh.css` / `app-zh.js`; English clutter and old
query-line media are in `site/archive/`. See `DATASET_CHOICE_ZH.md` for the newly verified
dataset/baseline rationale and `site/README.md` for the new build and browser commands.

- Updated: 2026-09-08 (America/New_York)
- Repository: `/home/hairo/pathrel_transfer/pathrel_pro6000`
- Durable checkpoint: `clean-ablation-evaluation-publication-20260908` in tracked `RECOVERY_STATE.json`
- Recovery-state commit: resolve with `git log -1 --format='%h %s' -- RECOVERY_STATE.json`
- Implementation history: `61d617e` is the September 2 base. The September 6 publication packages the clean-support implementation, audits and figures; resolve its revision with `git log -1 -- WORK_PLAN.md`. Current execution order is in `WORK_PLAN.md`.
- Scientific gate: **P0 GO; FlatLands K=128 clean-support validation candidate and bounded data gate
  historical numerical pass only; parent-place isolation failed on 2026-09-09; public-data test and paper claims remain gated**
- Handoff status: the clean FlatLands three-seed validation result is 0.06749 ± 0.00936
  versus independent 0.09521 ± 0.00703. The new nine-method paper table includes a strong
  same-checkpoint mean-map control (0.06957), exact fixed-marginal shuffle (0.16139), and
  equal-coverage risk analysis (3.60% versus 3.90% at 30%; all per-seed risk intervals include zero).
  The six-checkpoint nested K replay exactly reproduces every original K=128 prediction.
  UnScenes3D mean-map/qualitative v1 had a final-projection bug that reopened invalid support.
  Corrected v2 Brier is 0.51142 ± 0.00198 versus 0.51130 ± 0.00218; all 27,522 predictions
  satisfy the observation bounds and hidden-map metrics reproduce exactly. A frozen-observation
  lower bound of 0.47574 explains why this hard-observation adapter needs investigation before
  more training. PAPER_DRAFT.md and PAPER_EVIDENCE.md now reflect these positive and null results.
  All results remain validation-only. The physical test and UnScenes3D location_6 remain locked;
  do not extract the FlatLands archive. The next research gate is external-method/data compatibility
  as specified in EXPERIMENT_DESIGN_ZH.md. The no-event/no-global matrix is complete;
  its results and unchanged contract are in EVALUATION_SUMMARY_ZH.md. Observation-model and scalable-operator evidence remain required.

### GPU visibility and publication status (2026-08-31)

The full-access session exposes `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, and
`/dev/nvidia-modeset`; `nvidia-smi` sees an NVIDIA RTX PRO 6000 Blackwell (driver 580.173.02), and
the local `torch 2.13.0+cu130` venv reports `cuda_available=True, device_count=1, CC=12.0`.
Earlier no-GPU messages were a session/container passthrough failure, not physical GPU absence.
Training entry points still call `pathrel.gpu_diagnostics.cuda_unavailable_message` so future failures
remain layered and actionable.

After CUDA is visible, the reproducible six-run matrix (three seeds × ConPath / no-global / no-event-loss)
is launched with `scripts/run_flatlands_conpath_matrix.sh`. It writes an environment snapshot,
matrix manifest, per-run checkpoints, progress records, and external stdout logs; a partial run with
`latest.pt` is resumed, while ambiguous non-empty directories are refused.

`scripts/render_flatlands_qualitative.py` now renders a same-scene SVG from a real ConPath checkpoint
and the frozen FlatLands ZIP replay. It writes Observed BEV, ConPath posterior, optional deterministic
completion, and reference panels plus dataset/split/source/scene/query/radius/event metadata. It
refuses CUDA execution when the session cannot see a device and never fabricates a qualitative win.

The local `main` tree is clean and versioned; GitHub is synchronized through SSH over port 443. The
static site has been refreshed: its FlatLands section names the exact validation provenance
split/query/radii, highlights the best fixed baseline, and now lists the recent-paper bridge with
parameter matching and incompatibility status. Same-scene ConPath-versus-baseline map panels remain
reserved for a real checkpoint; no fabricated “ours is better” map is published.

`RECOVERY_STATE.json` is the machine-readable source of truth for required ignored artifacts,
byte counts, SHA-256 values, last verification, and the next action. Run the quick verifier first
after any interrupted session; run the full verifier when artifact integrity is in doubt. The
verifier is read-only and also prints the current Git status and the commit that last changed the
state file.

## Last completed work

The tracked repository already contains the TUM RGB-D geometric pilot, model hand-off smoke,
academic project page, exact NumPy merge-tree reference, and synthetic P0 baselines. The historical
neural failures below motivated the corrected two-seed result recorded later in this file.

The newest untracked/ignored run is `results/p0_neural_cuda_tuned01/` (generated 2026-08-28
06:01 local time). Its report records:

- 300 steps, 48 warm-up steps, 8 train samples, 128 validation samples;
- reachability weight 5.0 and categorical noise scale 0.1;
- event Brier `0.255772`, NLL `1.106823`, ECE `0.241109`;
- map-marginal Brier `0.007444`;
- event means by radius `[0.985555, 0.550191, 0.265706]` versus targets
  `[0.546875, 0.351562, 0.179688]`.

This is worse than the original 120-step neural run (`0.243578`) and the fixed P0 comparators:
independent Bernoulli `0.183172`, direct-query MLP `0.169888`. More generic step/weight tuning is
not the next action.

## Current diagnosis and implemented fix

The new `scripts/diagnose_p0_checkpoint.py` audit established that the tuned checkpoint is genuinely
over-open rather than merely noisy:

- true context-0/context-1 doorway-open rates are `0.21875/0.875`, while sampled rates are about
  `0.986/0.986`;
- about 30% of sampled doorway columns are fragmented, versus 0% in the targets;
- corrected unknown-region hard-map Brier is about `0.0949`; the old full-map soft score hid this;
- known visible evidence has zero violations.

Two concrete implementation faults were found and fixed in the current working tree:

1. with scaled Gumbel noise, the conditional categorical probabilities are
   `softmax(sample_logits / scale)`; the old code always used `softmax(sample_logits)`, so its
   reported/trained marginal did not match the hard-map generator when scale was not 1;
2. the old repeated straight-through extrema sent only about `2.7e-6` of an open-path closing
   gradient through the actual doorway. The model now uses exact hard events in the forward pass
   and the relaxed continuous maximum-bottleneck score only for the backward pass.

Joint training now scores only hidden cells, adds an empirical hard-map U-statistic Brier term, and
reports both conditional and empirical marginals plus doorway/context diagnostics.

## Interruption protection (verified)

`scripts/train_p0_neural.py` now writes one JSON record per step to `progress.jsonl`, atomically
saves `latest.pt` every `--checkpoint-every` steps, saves `interrupted.pt` on a caught failure or
Ctrl-C, preserves optimizer/model/Torch/CUDA/generator state, and refuses to overwrite an existing
run without `--resume`.

The recovery smoke ran steps 1-2, resumed the saved checkpoint, and completed steps 3-4 with the
loss history exactly `[1, 2, 3, 4]`. All tests currently pass: `Ran 35 tests ... OK`, `skipped=0`.

A 40-step 12x12 CPU regression completed without NaNs (event Brier `0.128662`, empirical full-map
Brier `0.020068`). It did not separate the two context priors and does not pass its reduced-protocol
direct-query baseline. This reduced setup has no positive radius-2 test events and is only a code
regression, not a scientific result.

## First official CUDA result after the gradient fix

`results/p0_neural_cuda_surrogate_v1/` completed 120 steps on the RTX PRO 6000. At 128 validation
samples it reports event Brier `0.163319`, ECE `0.017063`, and empirical full-map Brier `0.003747`.
Those aggregate scores beat independent (`0.183172`) and direct-query (`0.169888`). However, the
radius-zero predictions for context 0/1 are only `0.6027/0.6203`, versus targets `0.21875/0.875`;
the predicted gap recovers only about 2.7% of the target gap. Wall marginals, means, and factor
scales are likewise nearly context-invariant.

Therefore this result is **NO-GO despite its aggregate Brier**. The gate now additionally requires
recovering at least 50% of the held-out radius-zero context gap. The current working tree adds
coordinate channels and a globally pooled bottleneck context broadcast to the tiny P0 encoder;
the external `forward_features(...)` public-backbone path is unchanged.

The first CUDA evaluation-only resume also exposed and fixed a cross-device RNG restore bug:
serialized ByteTensor RNG states are explicitly moved back to CPU before `set_state`. The same
checkpoint then resumed successfully and completed the 128-sample evaluation.

`results/p0_neural_cuda_context_v2/` then tested that coordinate/global encoder for 120 steps. It
failed: event Brier `0.186264`, ECE `0.109511`, and context-gap ratio `0.0118`. A train-split
checkpoint diagnostic also showed almost identical context wall marginals (`0.1894/0.1909`), so
this is optimization/supervision collapse rather than held-out-template overfitting. The encoder
change alone is not a solution.

Replaying all training batch indices ruled out context sampling bias: joint-stage door rates were
`0.255/0.833` and full-train rates were `0.222/0.826`. The current working tree instead uses the
dataset's intended repeated-world structure: for each `(template, context)` it aggregates 24 worlds
into empirical map/event probability targets and applies proper scores to one shared observation.
The derived event still comes only from stochastic map samples. Legacy `--training-unit world`
remains available for old checkpoints.

`results/p0_neural_cuda_grouped_v3/` completed that grouped protocol. It reduced event Brier to
`0.169978` (direct-query is `0.169888`) and empirical hard-map Brier to `0.003525`, but the context
gap ratio remained only `0.0186`; the gate correctly failed. Grouping fixes target variance but not
the asymmetric conditional-input path.

The baseline evaluator and direct-query model receive the explicit visible `context` variable,
whereas the neural model so far had to infer it from the absolute position of an otherwise identical
2x2 landmark. The current working tree adds `--context-input plane`: a fourth, spatially broadcast
visible context-bit channel. It contains no doorway realization or event label. This gives the map
posterior the exact same conditioning variable as the baselines while retaining the original
three-channel `marker` mode for ablation/legacy checkpoints. All 35 tests pass; the official CUDA
run below is complete.

`results/p0_neural_cuda_contextplane_v4/` completed 120 steps with 128 validation samples and
**passed the tightened official gate**: event Brier `0.116377`, NLL `0.353872`, ECE `0.078647`,
empirical hard-map Brier `0.003383`, and false-safe@0.8 `0.125`. Radius-zero context predictions
were `0.4712/0.8476` versus targets `0.21875/0.875`, recovering `57.35%` of the target gap (minimum
required `50%`). This is a synthetic neural P0 result, not a public-data or paper result.

Residual weaknesses remain: context 0 is over-open and sampled doorway fragmentation is about
`13.7%/16.1%` for context 0/1 versus zero in the target.

`results/p0_neural_cuda_contextplane_noreach_v4/` is the matched ablation with only
`--reachability-weight 0` changed. It failed the gate: event Brier `0.191368`, NLL `0.795186`, ECE
`0.193570`, false-safe@0.8 `0.4258`, and context-gap ratio `0.2383`. Its empirical hard-map Brier
remained strong at `0.003098`, isolating the failure to joint/event structure rather than pixelwise
occupancy. Doorway fragmentation rose to `44.5%/71.6%`. Thus the reachability proper-score term is
material under this fixed seed: it improves event Brier by `0.074991` while the no-reach ablation
actually has slightly better marginal-map Brier.

`results/p0_neural_cuda_contextplane_seed20260828_v4/` completed the prescribed extra-seed run with
the dataset split fixed. It also passed: event Brier `0.111621`, NLL `0.346012`, ECE `0.071852`,
false-safe@0.8 `0.1125`, empirical hard-map Brier `0.002892`, and context-gap ratio `0.6957`.
Radius-zero context predictions improved to `0.3764/0.8329`. Fragmentation remained a limitation at
`16.1%/14.9%`, but it was far below the no-reach ablation's `44.5%/71.6%`.

Both full-model seeds independently pass every tightened gate; their event Brier range is
`0.111621-0.116377`, versus `0.169888` for direct-query and `0.183172` for independent Bernoulli.
The fixed-seed no-reach ablation fails. This closes the synthetic neural P0 gate, but it is not a
public-data or paper-level result.

All authoritative P0/readiness/roadmap/paper/transfer documents now record the same two-seed result,
matched ablation, synthetic-only boundary, and fragmentation limitation. The final regression run
reported `Ran 35 tests in 15.928s ... OK`, `skipped=0`; `smoke_forward.py` completed; and the 64x64
merge-tree reference matched exhaustive search at zero error for 8/64/512 queries (timing speedups
`3.09x/25.50x/161.19x`).

## P1 FlatLands archive audit

The workspace contains the ignored 830 MB TUM RGB-D pilot, the verified 2.055 GB FlatLands ZIP, and
the one-scene UnScenes3D raw pose smoke package plus the larger mini raw/label/local-map packages.
It has no ORFD or WildOcc assets. Official-source review selected FlatLands as the first
P1 audit target because it provides aligned observed/full floor maps, unobserved and valid masks,
metric provenance, and split metadata. ORFD remains a secondary off-road semantics audit; UnScenes3D
is now the priority second-domain candidate for support-surface occupancy.

`P1_DATA_AUDIT.md` freezes the FlatLands release as `2,054,773,316` bytes with SHA-256
`e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`, defines scene-disjointness,
mask/polarity, natural-query, 10%-15% event-balance, provenance/license, and strong-baseline gates.
The official split remains NO-GO. `scripts/download_flatlands.sh` supports a
resumable `.part`, verifies size and hash, atomically finalizes the archive, and never extracts it.
`bash -n` passes and `--check` now validates the present archive.

The committed `src/pathrel/flatlands.py`, `scripts/audit_flatlands_archive.py`, and tests implement a
read-only auditor for the frozen archive hash, unsafe/duplicate/symlink/encrypted members, five-file
packet completeness, official split counts, malformed metadata, source/scene identity, duplicate
global IDs, and cross-split scene leakage without extraction. Reports and manifests are written by
fsync plus atomic rename; progress/failure logs are fsynced incrementally.

The official archive is now present at
`data/raw/flatlands/FlatLands_final_dataset.zip`. Its exact size is `2,054,773,316` bytes and both
independent checks reproduce SHA-256
`e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f`. It has not been extracted.

The first bounded audit exposed the physical `val/` directory token; committed code normalizes it
to `validation`. The subsequent full scan in `results/p1_flatlands_archive_audit_full/` found all
270,575 complete packets and parseable metadata with zero unsafe/duplicate/symlink/encrypted/
unexpected members or missing identities. Official counts match exactly.

The decisive failure is scene leakage in the official observation-level split: train/validation
share 12,873 `(source, scene_id)` pairs, train/test 8,406, and validation/test 6,800. Every
in-distribution source leaks; ScanNet++ alone is test-only OOD. A concrete ZInD scene has 16/2/1
observations in train/validation/test. Therefore P1 remains **NO-GO on the official split**.

The full rerun from commit `cce7703` recovered a better split already encoded in every packet:
`provenance.original_split` preserves each upstream source's train/validation/test membership. It
contains 203,373/25,555/41,647 observations and 13,339/1,667/2,602 `(source, scene_id)` pairs, with
zero overlap, zero missing or unknown split values, and zero scenes assigned to multiple splits.
ScanNet++ contributes 16,214 observations only to provenance test. The deterministic 270,575-row
manifest is
`results/p1_flatlands_provenance_manifest/provenance_manifest.csv` (SHA-256
`a5eb28123f0fa2e38cc8244e6675c1eb76bc9a534ee54f56ef9ed68c4bdbc77b`). This is an explicitly
non-official FlatLands evaluation split, not a claim that the published archive directories pass.
Its pre-query integrity gate passes.

Implementation commits `8ee1c34` and `472c952` add a dependency-free grayscale PNG reader, stable
scene/observation sampling, target-blind metric polar queries, an exact linear-time disk-clearance
EDT, one-start multi-goal bottleneck scoring, atomic reports, and tests. The authoritative command
sampled 32 scenes per each of 16 provenance split/source strata (512 distinct scenes), one
observation per scene, and froze 36 queries per observation before reading `floor_map`. It read all
members directly from the ZIP and did not extract it.

The 512-observation mask audit found zero observed-floor/target disagreements, zero
observed/unobserved overlap, and zero invalid selected starts. It also quantified 5,415 observed and
22,539 target floor pixels outside `epistemic_mask`; the oracle explicitly excludes all such cells.
Of 18,432 query candidates, 6,735 passed target-blind selection, 2,082 then had target-invalid goals,
and 4,653 retained valid endpoints. Those contain 121 radius-zero disconnections, 3,095 larger-
footprint failures, and 1,437 20 cm positives.

Every validation/test source stratum passes the frozen minimum of 50 retained queries, eight
contributing scenes, and 0.10 scene-weighted disconnected/footprint rate; observed rates span
0.5823-1.0000. Thus the bounded mask/query data gate passes. It is not a calibration result and the
source/radius distribution is not uniformly balanced: test ARKitScenes has no 20 cm positives and
only 3/115 retained queries reachable at 10 cm. Future metrics must remain source/radius-stratified.
The reproducible result is `results/p1_flatlands_query_audit_bounded/`, generated from clean commit
`472c952`; report/selection/query SHA-256 values are registered in `RECOVERY_STATE.json`.

Implementation commit `a6ec796` completes the direct-ZIP hand-off in
`src/pathrel/flatlands_data.py`. It verifies the frozen CSV hashes and archive byte count, filters
only by `provenance.original_split`, lazily opens one ZIP handle per process, validates metadata and
masks, and reconstructs every query from input-side evidence before exposing targets. An all-packet
replay decoded 512/512 observations and reproduced split sizes 160/160/192, 4,653 retained queries,
and 10,452,053 valid hidden-region cells. All maps are 256x256; two-worker DataLoader and padded
query collation smokes pass. Tests reject archive-split fallback and tampered query geometry.

The same commit updates the project site with generated, tracked FlatLands audit data and figures:
`site/data/flatlands_audit.{json,js}`, `flatlands_reachability.svg`, and
`flatlands_query_outcomes.svg`. A real browser run populated 512 scenes, 4,653 retained queries,
11/11 gated strata, the three official overlap counts, and zero provenance overlap. The section is
labelled throughout as a bounded data audit and not a model/paper result. Full regression now reports
`Ran 52 tests in 1.104s ... OK`, `skipped=0`; `smoke_forward.py` also passes.

## P1 first validation baseline pass

The unified label-free evaluator, baseline runners, and canonical three-channel replay input are now
implemented. The evaluator joins targets only after exact prediction-key coverage is verified, uses
equal-scene primary weighting with 2,000-scene-cluster bootstrap intervals, and never reads the
locked test split during validation. The four validation-only methods and scene-weighted metrics are:

| Method | Brier | NLL | ECE | False-safe @0.8 | Coverage @0.8 |
|---|---:|---:|---:|---:|---:|
| Radius-prior control (train-only) | 0.15870 | 0.49283 | 0.01661 | 0.11156 | 0.33333 |
| Deterministic completion | **0.08556** | 1.18211 | 0.08556 | 0.08073 | 0.45450 |
| Independent-cell completion (K=32) | 0.22546 | 2.92768 | 0.24025 | **0.01454** | 0.20624 |
| Direct-query predictor | 0.09119 | **0.29788** | **0.04076** | 0.08335 | 0.37412 |

The deterministic completion has the lowest Brier; direct-query has the best NLL/ECE; independent
cells are a deliberately fragmented negative control with low false-safe coverage but poor event
calibration. These results establish evaluator plumbing and an initial difficulty baseline only. They
are not paper results: one seed, validation only, no official/public completion weights, and no second
domain yet. The direct-query run took 63.3 s (best epoch
42); marginal completion took 397.6 s including 353.4 s for K=32 event sampling on the RTX PRO 6000.

The project site now publishes `site/data/flatlands_baselines_validation.{json,js}` plus comparison
and reliability SVGs. Every public label says validation-only / test-locked / not a final paper result.

## First public-data ConPath pilot

The resumed `seed20260831_conpath` pilot completed on the RTX PRO 6000 after early stopping at
epoch 33 (best checkpoint at epoch 25). It used the historical low-capacity `F=8` encoder, 8
posterior worlds for training, 32 for validation selection, and the frozen provenance validation
split; the official archive split and test labels remained untouched. The exact-forward evaluator
produced 4,224 validation prediction rows with scene-weighted Brier `0.0836996`, NLL `0.385339`,
ECE `0.0550607`, false-safe@0.8 `0.0711530`, and zero radius-monotonicity violations. The
selection-time approximate event Brier was `0.0742096`. This is a single-seed validation pilot,
not a paper result, and is not directly capacity-matched to the existing `F=16` baselines.

The complete run manifest and predictions are retained under
`results/p1_flatlands_conpath_matrix/seed20260831_conpath/` (ignored artifacts); the next matrix
uses a separate output root with `F=16` and three fixed seeds. The matrix launcher now passes the
capacity argument into its manifest generator; recent strong methods remain required comparison
targets under the same contract before any final claim or site refresh.

## Capacity-matched F16 ConPath seeds

The first capacity-matched three-seed ConPath set is now complete under
`results/p1_flatlands_conpath_matrix_f16/`. All runs used the frozen provenance validation split,
8 training worlds, 32 validation worlds, exact-forward merge-tree evaluation, and no test labels:

| Seed | Best epoch | Selection Brier | Exact scene-weighted Brier | NLL | ECE | False-safe @0.8 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260831 | 4 | 0.08693 | 0.10809 | 0.64003 | 0.08514 | 0.04440 |
| 20260901 | 21 | 0.07559 | 0.08365 | 0.33534 | 0.05195 | 0.06595 |
| 20260902 | 10 | 0.08307 | 0.09647 | 0.39904 | 0.06902 | 0.06977 |
| **mean ± sample SD** | — | — | **0.09607 ± 0.01222** | **0.45814 ± 0.16071** | **0.06870 ± 0.01659** | **0.06004 ± 0.01367** |

These are still validation-only diagnostics, not final paper claims. The spread is material, and
the old F8 pilot (`0.08370` exact Brier) is not a fair capacity-matched comparison. The next
required step is the matched causal ablation set (no global factors, no event proper score,
independent decoder, deterministic mean map, and K convergence), followed by source/radius
uncertainty and recent strong-method comparisons under this same contract.

## F16 causal ablations completed

The two implemented causal ablations are now complete for all three fixed seeds, with exact
forward validation on the same 4,224-row-per-seed manifest:

| Variant | Exact scene-weighted Brier (20260831 / 20260901 / 20260902) | Mean ± sample SD | NLL mean ± SD | ECE mean ± SD |
|---|---:|---:|---:|---:|
| ConPath | 0.10809 / 0.08365 / 0.09647 | **0.09607 ± 0.01222** | 0.45814 ± 0.16071 | 0.06870 ± 0.01659 |
| No global factors | 0.12134 / 0.10812 / 0.11568 | 0.11505 ± 0.00663 | 0.95790 ± 0.06133 | 0.09649 ± 0.01168 |
| No reachability loss | 0.20048 / 0.20983 / 0.20635 | 0.20555 ± 0.00473 | 2.57912 ± 0.08818 | 0.21286 ± 0.01620 |

Removing global factors worsens mean Brier by `0.01898` and NLL by `0.49976`; removing the
reachability proper score worsens mean Brier by `0.10948` and NLL by `2.12098`. These are strong
mechanistic signals, not final paper claims: all rows are validation-only, and false-safe/coverage
 trade-offs plus source/radius bootstrap intervals remain required. The independent decoder,
deterministic mean-map, K-convergence, and PaSCo-inspired controls are now complete; additional
recent-method ports remain pending.

## Scalable connectivity checkpoint

The exact NumPy Kruskal reconstruction-tree oracle is now exposed as
`batched_merge_tree_bottleneck_scores` for `[B,K,H,W]` maps and `[B,Q,2]` query sets. It builds one
tree per map and answers all terminals with LCA lookups, so query cost is logarithmic after the map
build rather than another `H*W` relaxation per query. A strict test compares every batch/sample/query
entry to the existing single-map oracle. The synthetic CPU contract benchmark
`results/p1_flatlands_connectivity_benchmark/benchmark.json` uses 64×64 maps and K=4: 32/128/512/2048
queries take about 0.036/0.037/0.044/0.060 s, respectively. A larger shape check at 256×256 maps,
B=2, K=8 (`results/p1_flatlands_connectivity_benchmark/benchmark_b256_k8.json`) keeps the exact
output finite and takes 2.75/2.75/2.85/2.91/3.03 s for 32/128/512/2048/4224 queries. The near-flat
query-time growth after tree construction confirms that batching amortizes the expensive map build.
This is an exact-forward reference and efficiency diagnostic only; a CUDA implementation and soft
backward path are still pending.

`scripts/train_flatlands_conpath.py` is now the reproducible public-data neural entry point. It uses
the canonical replay channels, hidden-cell posterior NLL, variogram score, reachability U-statistic,
shared-start propagation, atomic checkpoints, and label-free validation predictions. Its final
validation path converts hard posterior worlds to exact disk-clearance maps and evaluates all queries
with the batched Kruskal merge-tree, while the bounded differentiable propagation is used only for
training/epoch selection. A one-scene CPU smoke completed end to end (including checkpoint/prediction
writing) with `paper_result=false` and validation subset evaluation disabled. Full validation runs
must still use the frozen 160-scene provenance split and multiple seeds before becoming paper results.

## Completion multi-seed controls and K convergence

Two additional completion seeds were trained under the frozen P1 protocol. They complete the
three-seed control set alongside `seed20260831`; all remain validation-only and test-locked:

| Seed | Best epoch | Map Brier | Map NLL | Deterministic event Brier | Independent K=32 event Brier |
|---:|---:|---:|---:|---:|---:|
| 20260831 | 15 | 0.11345 | — | 0.08556 | 0.22546 |
| 20260901 | 26 | 0.11640 | 0.37618 | 0.08816 | 0.23187 |
| 20260902 | 13 | 0.11450 | 0.36853 | 0.09199 | 0.23611 |

For seed `20260901`, one fixed completion checkpoint was evaluated with a shared posterior stream
at K=32/64/128. Event Brier was `0.231873/0.231068/0.230949`, NLL was
`2.992125/2.957632/2.936882`, and ECE was `0.248903/0.248603/0.248616`. The small change confirms
that K=32 is adequate for rapid diagnostics; final tables retain K=128. The evaluation script now
generates the maximum K once and reports all prefixes, so this convergence check is reproducible
without three independent expensive samplers.

## Recent-paper same-contract control

The first recent-method adaptation is a PaSCo-inspired three-subnet ensemble control. It uses the
three completion checkpoints above, the frozen three-channel FlatLands input, and a fixed total of
32 event worlds (11/11/10 per member). It is explicitly not a reproduction of PaSCo's camera/3-D
architecture. The exact scene-weighted validation result is map Brier `0.110545`, event Brier
`0.228984`, NLL `2.898173`, ECE `0.248410`, and false-safe@0.8 `0.015380`. The compact report and
member/prediction hashes are tracked in `site/data/flatlands_pasco_ensemble_validation.json`; the
full ignored artifacts remain under `results/p1_flatlands_pasco_ensemble_validation/`.

### S4C-inspired coordinate-query control (completed)

`S4CInspiredCoordinateBaseline` is now available through
`scripts/train_flatlands_direct_query.py --architecture s4c_coordinate`. It keeps the canonical
three-channel replay, F=16 encoder, frozen train/validation scenes, 4,224 retained event rows,
radii 0/10/20, AdamW protocol, and test lock. The only architectural change is a coordinate-query
implicit field: bilinear start/goal feature sampling plus Fourier encodings of geometry and radius.
It is explicitly a recent-method-inspired control, not a reproduction of S4C's original 3-D system.

Three seeds completed in parallel without OOM or protocol failures:

| Seed | Best epoch | Exact scene-weighted Brier | NLL | ECE | False-safe @0.8 |
|---:|---:|---:|---:|---:|---:|
| 20260831 | 18 | 0.09873 | 0.34064 | 0.04406 | 0.09431 |
| 20260901 | 38 | 0.08812 | 0.34856 | 0.05047 | 0.09958 |
| 20260902 | 21 | 0.08927 | 0.31314 | 0.03452 | 0.09276 |
| **mean ± sample SD** | — | **0.09204 ± 0.00582** | **0.33411 ± 0.01859** | **0.04302 ± 0.00803** | **0.09555 ± 0.00357** |

The coordinate control is competitive on event Brier/NLL/ECE, but its false-safe risk at the 0.8
threshold is higher than ConPath (`0.09555` versus `0.06004`). The four-method calibration curves
and JSON snapshot are regenerated on the site; all numbers remain validation diagnostics and do not
support a final paper or SOTA claim.

### Official FlatLands artifact check (completed)

The official [FlatLands project page](https://1ssb.github.io/Flat_Lands/),
[GitHub repository](https://github.com/1ssb/Flat_Lands/), and
[Hugging Face dataset card](https://huggingface.co/datasets/Rudra1ssb/FlatLands) were checked on
2026-09-02. The validated archive and documentation are public, but the official repository states
that model weights, construction code, and additional benchmark tooling are planned for release.
There is no importable official checkpoint/evaluator to run under the ConPath three-channel,
256×256, provenance-original split, and exact event contract. The official method therefore remains
reference-only; the detailed compatibility checklist is tracked in `OFFICIAL_FLATLANDS_CHECK.md`.

### SceneSense diffusion compatibility check (completed)

The official [SceneSense repository](https://github.com/arpg/SceneSense) and its IROS/online papers
were checked on 2026-09-02. SceneSense is a 3-D point-cloud/voxel ROS inpainting system using a
running robot map; its public inputs and FID/KID/exploration metrics do not match the FlatLands
three-channel 2-D packet or ConPath event metrics. No checkpoint/preprocessing path is available for
a faithful same-contract run. It is therefore reference-only, with details in
`SCENESENSE_COMPATIBILITY_CHECK.md`; a new 2-D diffusion adapter is deferred rather than mislabeled
as a SceneSense reproduction.

### ORFD second-domain compatibility check (completed for planning)

The official [ORFD paper](https://arxiv.org/abs/2206.09907) and [implementation/data repository](https://github.com/chaytonmin/Off-Road-Freespace-Detection)
were checked on 2026-09-02. ORFD is ground-vehicle off-road RGB/LiDAR data with 12,198 pairs from
30 sequences and pixel-wise `traversable`/`non-traversable`/`unreachable` image labels. It is a
stronger semantic candidate for the replacement visual than the desk pilot, but its task is not
our metric hidden-grid two-terminal event: the paper labels the image plane, merges `unreachable`
into `non-traversable` for evaluation, and only documents pair-level train/validation/test counts.
The official loader uses per-frame camera intrinsics for depth/normal processing but does not expose
an ego-pose/world-map stream. No sequence-held-out guarantee or world-frame support-map/pose
artifact has been established for our use. ORFD is therefore recorded as **candidate secondary
domain; semantics audit complete, no local run**. The required download/hash, calibration/pose,
leakage, metric-BEV adapter, and label-validity gates are tracked in
`ORFD_COMPATIBILITY_CHECK.md`.

### UnScenes3D second-domain compatibility check and adapter (validation controls complete; final/test score pending)

The official [UnScenes3D repository](https://github.com/ruiqi-song/UnScenes3D), its [release page](https://github.com/ruiqi-song/UnScenes3D/releases),
and the [Scientific Data article](https://www.nature.com/articles/s41597-025-05532-5) were checked on
2026-09-02. The release provides 3-D semantic occupancy, road elevation, local dense maps, camera/
LiDAR calibration, and vehicle/ego-pose information; the article reports camera/LiDAR/IMU/RTK
collection and approximately 23,549 frames. The official processing code loads `Tr_velo_to_imu`,
`pose_odom`, local-map clouds, and scene timestamps, so it is a viable metric-BEV source in
principle. The six-region description also provides an unseen-region generalization design.

The official `raw_data.zip` mini asset (104,519,506 bytes,
SHA-256 `d1050b22d0eb31ea7199d2625accbfb54ae3034b3b57653a9790cdbdfa522ae0`) has been downloaded,
path-audited, and extracted under ignored `data/raw/unscenes3d/raw_data/`. The larger mini raw
package has 13 scenes and 1,336 synchronized image/cloud/calibration stems; the label package has
629 aligned occupancy/elevation/depth timestamps. Both local-map parts are now also present: their
629 files cover all 629 occupancy timestamps. The read-only mini audit passes zero missing raw
stems, duplicate timestamps, occupancy shape/bound/class errors, and bad map floats; a 50-frame
scene-spread nearest-neighbour smoke gives 0.804 LiDAR/map overlap within 0.3 m and 0.875 within
1 m. These are parser/label/coordinate diagnostics, not experiments. The coordinate audit now
covers 55 train/validation frames with positive-depth camera projection for all 800,742 sampled
returns, calibration rotation error at most `2.13e-6`, and direct LiDAR/local-map overlap of
0.791 within 0.3 m and 0.863 within 1 m. Both forward and inverse `Tr_velo_to_imu` transforms
had zero overlap at those thresholds, so the adapter uses the raw LiDAR frame directly for this
local-map contract. `UNSCENES3D_PROTOCOL.md` v0.2 freezes the location-held-out train/validation
(`location_1/2/3` vs `location_4_5`) with `location_6` test locked, 0.3 m grid radii 0/1/2, and
a fixed 13/27/40-cell polar stencil. `src/pathrel/unscenes3d.py` implements the label-free LiDAR
ray rasterizer, conservative class-11 support projection, and validity-only query geometry; six
adapter tests pass. The canonical candidate uses a label-free ground-height endpoint policy
(1.2 m bins, 20 m lateral limit, 0.15 quantile, 0.35 m margin) while retaining the fixed
validity-only start. Its manifest at
`results/unscenes3d_contract_manifest_ground_valid/manifest.json` contains the same 17,096 exact
event queries (15,567 train and 1,529 validation) as the legacy manifest, and a 540-frame runtime
replay found zero start/goal mismatches. The observed-free-start variant changed the query set and
was rejected as the canonical contract.

The first full legacy capacity-matched map-only adapter run covered all 478 train and 62 validation
frames for seeds `20260831`, `20260901`, and `20260902`; its validation event Brier was
`0.63165 ± 0.00083`, with NLL `10.06875 ± 0.02700` and ECE `0.73653 ± 0.00143`. The bounded
event-loss variant was `0.63704 ± 0.00317`, exposing a mismatch between the differentiable path
used for optimization and the exact oracle used for reporting. On the canonical ground-valid
contract, the three-seed map-only diagnostic improved to event Brier `0.56272 ± 0.00744` (NLL
`9.06574 ± 0.04398`, ECE `0.67082 ± 0.00739`, false-safe@0.8 `0.27450 ± 0.01330`). The earlier
five-epoch bounded event-loss number `0.59126 ± 0.00855` (NLL `9.19562 ± 0.06403`) is an
observed-free-start ablation and is not a canonical-contract result. Both remain adapter
diagnostics, not method wins or paper results.

## UnScenes3D controls and ground-robot visual (2026-09-02)

The validation-only deterministic mean-map evaluator is now in
`scripts/evaluate_unscenes3d_conpath_mean_map.py`. It reads the manifest adapter, checks every
replayed start/goal against the frozen query rows, averages Rao--Blackwellized posterior marginals,
and runs the exact clearance oracle on one thresholded map; it never opens test records. The
canonical correlated K=128 mean-map control across the three checkpoints is event Brier
`0.62774 ± 0.00145`, hidden-cell map Brier `0.15025 ± 0.03856`, NLL `9.96833 ± 0.01217`,
ECE `0.72153 ± 0.00088`, and false-safe@0.8 `0.39940 ± 0.00182`. K sensitivity for the fixed
seed/checkpoint is hidden-map Brier `0.19399`/`0.19385`/`0.19391` at K=32/64/128, with the
binary event Brier fixed at `0.62750`.

The capacity-matched independent decoder (`--decoder-variant independent`) completed on the
same contract: stochastic event Brier `0.56297 ± 0.00541`, NLL `9.08667 ± 0.04961`, ECE
`0.67240 ± 0.00525`, false-safe@0.8 `0.27027 ± 0.01767`. Its K=128 mean-map event Brier is
`0.62762 ± 0.00126`; the identical binary output to displayed precision is itself a useful
warning that the current adapter/threshold dominates this control.

The new `scripts/train_unscenes3d_coordinate.py` trains an explicitly labelled S4C-inspired
coordinate-query control (not an S4C reproduction) with F=16, the same four-frame batches,
AdamW budget, query rows, and site-held-out validation. After correcting the scene-weighting
definition and rerunning from scratch, the three seeds in
`results/unscenes3d_s4c_coordinate_f16_v2/` report event Brier `0.20379 ± 0.03485`, NLL
`0.57289 ± 0.10572`, ECE `0.11107 ± 0.05312`, false-safe@0.8 `0.13342 ± 0.01875`, and zero
radius-monotonicity violations. This direct event predictor is a control with a different
predictive object; it is not evidence of ConPath superiority or inferiority.

`scripts/render_unscenes3d_qualitative.py` scanned all 62 validation frames with K=128 and wrote
the audited positive/false-safe pair under `results/unscenes3d_qualitative_validation/`:
`scene_00427/1693304934.305044` is an all-radii true positive, while
`scene_00427/1693304969.384475` is a deliberate all-radii false-safe case (predicted reachable,
target blocked). The 1800×1280 PNGs are mirrored to `site/assets/` and the selection/hash report
to `site/data/unscenes3d_qualitative_validation.json`; Pillow was used only for rendering, and
the model scan used the CUDA environment. The site now has an UnScenes3D section with the contract,
control table, K sensitivity, and both panels. `location_6` remains locked.

### Overnight audit checkpoint (executed)

The first control audit found stale report-envelope metadata rather than a metric or geometry
failure: the original correlated JSONs carry a v0.1 protocol tag, while their checkpoint configs
already use the frozen ground-valid adapter. I did not overwrite those raw reports. Instead, the
current training script now emits a self-contained manifest hash/query envelope, and a fresh
three-seed replay was written under `results/unscenes3d_ground_valid_f16_v2/`. The replay has the
same adapter, seed/config, validation-only lock, and event metrics to within `1.51e-4` absolute
(zero drift for two seeds); this is far below the displayed precision and is recorded by
`scripts/compare_unscenes3d_replay.py` in `results/unscenes3d_replay_audit/report.json`.

The read-only `scripts/audit_unscenes3d_controls.py` then recomputed every label-free prediction
CSV against the frozen 4,587 validation rows, checked event Brier/NLL/ECE/false-safe values,
radius ordering, checkpoint/prediction hashes, K reports, the radius prior, qualitative PNG
dimensions, manifest-replay contract, published evidence copies, portable published paths, and all local website links: **599 checks passed, 0 failed**. Its only three warnings
are the intentionally preserved v0.1 metadata tags on the original correlated reports; the v0.2
replay is the metadata-stable cross-check. `scripts/compare_unscenes3d_manifest.py` independently
rebuilt the ground-valid manifest from the explicit train/validation allow-list and matched every
contract field; its normalized contract SHA is `017b368b0364ca518e1c0d744ed81d53d92f190b72ee66109417b29c5bee77d5`.
The qualitative report now includes its 1,529-query
count and the promoted site JSON was refreshed without changing either PNG byte hash.

To close the selective-risk block, `scripts/evaluate_unscenes3d_calibration.py` joined the
label-free mean-map and coordinate-query CSVs (plus a train-scene/frame-fitted radius prior) to the
same validation keys and generated query-weighted reliability and false-safe curves. The snapshot
is `site/data/unscenes3d_calibration_validation.json`, with SVGs
`site/assets/unscenes3d_calibration_reliability.svg` and
`site/assets/unscenes3d_calibration_false_safe.svg`. The scalar radius-prior check reproduces
probabilities `0.87174096/0.81066486/0.75479295` and Brier/NLL/ECE/false-safe
`0.23268/0.52509/0.03237/0.19359`; the chart caption explicitly distinguishes query-weighted
risk from the table's equal-scene Brier. The stochastic controls remain scalar-only in this view,
so no fabricated curve is shown for them. Both SVGs were rasterized in headless Chrome after a
layout pass that separates the legend from the axis labels; the ground-robot PNGs were inspected
at their native 1800×1280 dimensions.

## Autonomous overnight plan (minimum 13 hours)

This is the recorded long-running plan requested on 2026-09-02. Each block ends with a read-only
audit and a durable state update; a failed gate causes a repair/re-run or a documented hold rather
than silently advancing.

| Window | Work | Exit evidence / branch rule |
|---|---|---|
| 0–1 h | Freeze the v0.2 manifest, adapter parameters, checkpoint list, environment versions, and test lock. | Manifest/runtime starts and goals match for every train/validation frame; if not, rebuild the manifest and invalidate dependent scores. |
| 1–3 h | Re-run deterministic mean-map K=32/64/128 and independent K=128 controls from fixed seeds; compute scene/site-weighted Brier, NLL, ECE, false-safe risk, coverage, and radius monotonicity. | K curves stable and reports contain no test paths; otherwise fix evaluator semantics before any site update. |
| 3–5 h | Audit the S4C-inspired coordinate-query runs, prediction row order, checkpoint hashes, and seed dispersion; compare against a radius prior and the correlated row without ranking unlike predictive objects as one winner. | Exact replay of all 4,587 validation rows per seed and zero geometry mismatches; high variance is reported, not hidden. |
| 5–7 h | Inspect and, if needed, re-render the ground-robot positive and false-safe panels at native resolution; verify camera/LiDAR projection, BEV orientation, captions, and target-selection disclosure. | Two readable PNGs plus JSON selection/hash report; no target-derived image is presented as an input. |
| 7–9 h | Build compact site snapshots and integrate the UnScenes3D section; run HTML asset/link checks and a local static-site smoke. | All linked assets exist, metrics match JSON to displayed precision, and the site says validation-only/location_6 locked. |
| 9–11 h | Run the full test suite, compileall, `git diff --check`, protocol consistency checks, and quick recovery verification. | Zero failures; any mismatch is repaired before proceeding. |
| 11–13 h | Run the full recovery/hash verifier if I/O budget permits, record artifact hashes and environment metadata, and prepare a reproducible morning hand-off. | `RECOVERY_STATE.json` and this file name exact commands, outputs, and unresolved holds. |
| 13 h+ | Continue paper-package drafting, sensitivity/failure analysis, and optional non-test ablations (e.g. threshold/K checks) only if all prior gates pass. | Never read `location_6`; stop at the explicit test go/no-go boundary and leave a resumable state. |

The table is a minimum 13-hour work window for an unattended session, not a claim that the checks
consume 13 hours of wall time. In this run the bounded experiments and audits completed ahead of
that schedule; the exact outputs, hashes, and the remaining explicit test boundary are recorded
below so a later session can resume without inventing elapsed work.

The standing invariants for every block are: no extraction or use of the FlatLands archive, no read of
UnScenes3D `location_6`, no claim stronger than the recorded validation contract, atomic writes for
checkpoints/reports, and an independent verification command before a result is promoted into the
website or paper draft.

Execution ledger for this hand-off: the manifest/geometry freeze, deterministic manifest replay,
K controls, three-seed coordinate audit, GPU replay comparison, native-resolution panel check,
compact site/static-link and calibration snapshot checks, published audit evidence copies,
paper/site consistency check, unit tests, compileall, `git diff --check`, quick recovery
verification, both FlatLands K=128 decoder audits, and the full SHA verifier have all passed. The
full verifier now checks 344 state entries (including the large archives, both clean K=128
and the paired reports/candidate panels) with zero failures. The
only open note is the three intentionally preserved v0.1 metadata warnings on the legacy correlated
JSONs; the v0.2 replay is the metadata-stable hand-off. No split, query, target, or test lock was
changed to obtain these results. A later site-link refresh briefly made the published audit copy
stale; regenerating the source report and then copying it restored byte equality and 599/599 checks.

## Dataset readiness checkpoint (2026-09-03)

The local-data readiness check confirms that the current paper package has its raw inputs: the
FlatLands release archive is present with the frozen release hash; UnScenes3D has 13 scenes, 629
joined timestamps, 629 local maps, and a passing basic-integrity audit; and the TUM RGB-D desk RGB/depth
pilot is extracted. The full recovery verifier was rerun with hashes enabled and passed with zero
failures. This verifies file availability, not scientific finality: FlatLands remains a non-official
provenance split because the published split leaks scenes, and UnScenes3D `location_6` remains
unopened. The recommended paper branch is therefore P0 synthetic + FlatLands primary, with
UnScenes3D as a separately labelled secondary diagnostic; ORFD/WildOcc are not local and are
optional future additions.

The next training/test branch is deliberately held at the protocol boundary: preserve the primary
dataset and hyperparameters, review the completed matched validation-only controls, and request an
explicit go/no-go before reading a locked test site. No test labels were accessed during this
readiness check.

## Validation qualitative panels and website state

Two real-checkpoint FlatLands validation panels were rendered and copied to the tracked site:
`flatlands_qualitative_validation_positive.svg` (r=10, ConPath event `0.844`) and
`flatlands_qualitative_validation_uncertain.svg` (disconnected r=0 case, ConPath event `0.563`).
Each panel shows observed BEV, ConPath posterior, deterministic completion, and the reference map;
the second case is intentionally included to avoid a one-sided visual claim. The current TUM video
is still a geometry/BEV pilot. A final paper/site video must show posterior updates, radius-
conditioned reachability, baseline comparison, and at least one failure/overconfidence case.

### Visual presentation requirement (new)

The desk-surface BEV is not semantically appropriate as the main mobile-robot navigation visual.
It remains only a clearly-labelled geometry-pipeline pilot until a ground-robot/floor sequence is
available. The hero video and paper figures must be large enough to read on first view, use no more
than a few panels per figure, keep a shared legend/scale, and explain the chain
`partial observation → posterior worlds → footprint erosion → path probability`. Existing dense
multi-panel SVGs should be treated as diagnostics/appendix material until they are replaced or
re-rendered with this visual hierarchy.

## Independent-decoder control (completed)

The independent-cell decoder control has been added to
`scripts/train_flatlands_conpath.py` as `--decoder-variant independent`. It uses the same
three-channel replay input, optimizer, train/validation scenes, K=32 validation worlds, exact
forward evaluator, and three fixed seeds as the F16 ConPath matrix, while removing the correlated
local/global posterior structure (`local_kernel_size=1`, global factors disabled). It is a causal
control, not a recent-paper reproduction and not a final paper claim.

To use all available GPU capacity without the previous validation OOM, validation K=32 is now
processed as four sequential K=8 chunks (`--validation-sample-chunk 8`) and accumulated with the
same total sample count. The three seeds are running concurrently with atomic latest/best
checkpoints under `results/p1_flatlands_conpath_independent_f16/`; interrupted checkpoints are
resume-safe. The three exact validation manifests and reports are now complete (4,224 rows per
seed):

| Seed | Best epoch | Selection Brier | Exact scene-weighted Brier | NLL | ECE | False-safe @0.8 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260831 | 4 | 0.08522 | 0.11490 | 0.95813 | 0.10288 | 0.07114 |
| 20260901 | 10 | 0.08047 | 0.10884 | 0.90150 | 0.08816 | 0.07314 |
| 20260902 | 10 | 0.08627 | 0.11267 | 0.95242 | 0.09282 | 0.07629 |
| **mean ± sample SD** | — | — | **0.11214 ± 0.00306** | **0.93735 ± 0.03118** | **0.09462 ± 0.00752** | **0.07352 ± 0.00260** |

The matched F16 ConPath row remains lower at `0.09607 ± 0.01222` Brier, so the independent
decoder is a causal control rather than evidence that local correlation is unnecessary. The run
manifests record peak memory and wall time; all three use the same test-locked validation contract.
This control currently exports event predictions only; a separate deterministic mean-map control is
still required before any map-quality claim is made.

## Deterministic posterior mean-map control (completed)

The new `scripts/evaluate_flatlands_conpath_mean_map.py` evaluator averages 128 conditional
posterior free probabilities from each F16 ConPath checkpoint in K=32 chunks, thresholds the mean
map at 0.5, and runs the exact connectivity oracle on that single binary map. It also scores hidden
map probabilities before thresholding. All three seeds completed on the same 160-scene validation
replay and 4,224 event rows:

| Seed | Event Brier | Event NLL | Event ECE | Hidden-map Brier | Hidden-map NLL | Mean map probability |
|---:|---:|---:|---:|---:|---:|---:|
| 20260831 | 0.07512 | 1.03787 | 0.07512 | 0.17679 | 0.58670 | 0.79509 |
| 20260901 | 0.07058 | 0.97506 | 0.07058 | 0.17431 | 0.55942 | 0.80110 |
| 20260902 | 0.06763 | 0.93439 | 0.06763 | 0.18416 | 0.65597 | 0.80557 |
| **mean ± sample SD** | **0.07111 ± 0.00377** | **0.98244 ± 0.05214** | **0.07111 ± 0.00377** | **0.17842 ± 0.00512** | **0.60070 ± 0.04977** | **0.80059 ± 0.00525** |

The lower binary event Brier is not a free win: hidden-map Brier is substantially worse than the
stochastic posterior, event calibration collapses to a 0/1 output, and false-safe@0.8 averages
`0.11740`. This validates retaining the stochastic posterior and records the event/map trade-off
for the paper's ablation section. The report and compact site snapshot remain validation-only.

## Uncertainty and selective-risk curves (completed)

`scripts/evaluate_flatlands_calibration.py` joins the nine existing label-free validation
prediction manifests to the frozen query labels and aggregates equal-scene reliability and
high-confidence risk across seeds 20260831/20260901/20260902. It does not train or open the test
split. The compact snapshot is `results/p1_flatlands_calibration_validation/calibration_snapshot.json`
and the public copies are `site/data/flatlands_calibration_validation.json`,
`site/assets/flatlands_calibration_reliability.svg`, and
`site/assets/flatlands_calibration_false_safe.svg`.

At the 0.8 confidence threshold, ConPath has false-safe rate `0.06004 ± 0.01367` and accepted-event
coverage `0.29114 ± 0.01591`; the independent decoder has `0.07352 ± 0.00260` false-safe and
`0.34447 ± 0.00852` coverage. The deterministic mean-map control accepts more events (`0.51697`)
but is binary and has `0.11740 ± 0.00584` false-safe. The reliability diagram and threshold sweep
make this calibration/coverage trade-off visible instead of ranking methods by event Brier alone.
All values are validation diagnostics, not final paper or test results.

The calibration snapshot also records the nested-radius invariant. For every completed control and
every seed, `p(r=0) >= p(r=10) >= p(r=20)` has zero violations over 1,408 endpoint groups (2,816
adjacent-radius pairs); the joined validation labels likewise have zero violations. This confirms
that the event sampler preserves the footprint ordering, but it is not a symmetry or generalization
claim.

## PaSCo-inspired sample-budget check (completed)

The existing three-subnet same-contract control was rerun with total posterior event budgets
`K=32/64/128` (member allocations remain as even as possible). Exact scene-weighted validation
metrics are:

| Total K | Event Brier | Event NLL | Event ECE | False-safe @0.8 | Coverage @0.8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.22898 | 2.89817 | 0.24841 | 0.01538 | 0.18981 |
| 64 | 0.22842 | 2.86991 | 0.24810 | 0.01465 | 0.18691 |
| 128 | 0.22806 | 2.84268 | 0.24830 | 0.01228 | 0.19122 |

The small changes show that this ensemble-control result is not an artifact of K=32 sampling.
The K=64 and K=128 predictions and exact reports remain ignored reproducibility artifacts under
`results/p1_flatlands_pasco_ensemble_validation/`; the compact K-sensitivity snapshot is tracked
on the site. The adapter is still explicitly PaSCo-inspired, not an original PaSCo 3-D
reproduction, and all numbers remain validation-only.

## FlatLands K=128 ConPath validation candidate (completed 2026-09-03)

To match the paper protocol's final posterior budget, the frozen F=16 ConPath contract was rerun
for seeds 20260831, 20260901, and 20260902 with `validation_samples=128` accumulated as
K=8 chunks. The run used the same 160 train / 160 validation provenance scenes, 8 training
worlds, query limit 8, radii 0/10/20, exact NumPy disk-clearance plus batched Kruskal/LCA forward,
and 2,000 scene-cluster bootstrap replicates. It never opened the physical archive split's test
labels; every `run.json` records `paper_result=false` and `test_evaluated=false`.

| Seed | Best epoch | Epochs | Selection Brier | Exact validation Brier | NLL | ECE | False-safe @0.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260831 | 4 | 12 | 0.08558 | 0.10465 | 0.49132 | 0.08680 | 0.04299 |
| 20260901 | 16 | 24 | 0.07221 | 0.08285 | 0.40156 | 0.05628 | 0.05648 |
| 20260902 | 8 | 16 | 0.08236 | 0.10000 | 0.39265 | 0.06794 | 0.06611 |
| **mean ± sample SD** | — | — | — | **0.09583 ± 0.01148** | **0.42851 ± 0.05458** | **0.07034 ± 0.01540** | **0.05519 ± 0.01161** |

High-confidence accepted-event coverage is `0.28974 ± 0.02783`. The K=32 F=16 matrix was
`0.09607 ± 0.01222` Brier, `0.45814 ± 0.16071` NLL, and `0.06870 ± 0.01659` ECE, so the
K=128 replay is protocol-stable within the three-seed dispersion (and modestly lowers the mean
Brier/NLL), rather than a new unqualified claim. The selection score is stochastic because it
uses posterior worlds; final comparisons therefore use the independently written exact
validation manifests, not the lowest intermediate epoch log.

All three runs wrote 4,224 finite `[0,1]` prediction probabilities over 1,408 endpoint groups;
the recorded SHA-256 values match the files, the independent evaluator replay reproduces each
report, and radius ordering `p(0) >= p(10) >= p(20)` has zero violations. The compact machine-
readable hand-off is `site/data/flatlands_conpath_k128_validation.json`; the full ignored run
directories are under `results/p1_flatlands_conpath_k128_validation_v2/`. This is a **validation
candidate only**: the non-official FlatLands provenance caveat, locked test boundary, and explicit
go/no-go requirement remain unchanged.

The three selected `best.pt` files load cleanly into the current `PathRelNet` with no missing or
unexpected state keys and `120,108` trainable parameters. The matched independent-decoder
checkpoints have the same trainable count; their control difference is the decoder behavior
(`local_kernel_size=1` and effective global factors disabled), not a hidden capacity change.

## Matched FlatLands independent-decoder K=128 control (completed 2026-09-03)

To complete the capacity- and sample-matched causal comparison, three independent-cell decoder
runs were executed under `results/p1_flatlands_independent_k128_validation_v2/` for seeds
`20260831`, `20260901`, and `20260902`. They use the identical F=16 encoder, 160/160 provenance
train/validation scenes, eight training worlds, 128 validation worlds in K=8 chunks, query limit,
radii, optimizer, exact-forward evaluator, and bootstrap budget as the ConPath candidate; only the
correlated local/global posterior structure is removed (`decoder_variant=independent`, effective
`local_kernel_size=1`). The explicit `--disable-global-factors` flag remains false because the
decoder variant itself applies the ablation. This is a validation-only control and did not open the
FlatLands physical test split.

Training stopped by the fixed early-stopping rule (best epochs: seed 20260831 → 4 of 12, seeds
20260901/02 → 10 of 18), followed by exact validation manifest generation. The final scene-weighted
metrics are:

| Seed | Exact Brier | NLL | ECE | False-safe @0.8 | Coverage @0.8 |
|---:|---:|---:|---:|---:|---:|
| 20260831 | 0.11476 | 0.86451 | 0.09921 | 0.07882 | 0.34011 |
| 20260901 | 0.11032 | 0.85464 | 0.08716 | 0.07130 | 0.35626 |
| 20260902 | 0.11673 | 0.91047 | 0.10507 | 0.08560 | 0.34424 |
| **mean ± sample SD** | **0.11394 ± 0.00328** | **0.87654 ± 0.02979** | **0.09715 ± 0.00913** | **0.07857 ± 0.00715** | **0.34687 ± 0.00839** |

All three manifests contain 4,224 finite rows over 1,408 endpoint groups, with zero radius-order
violations. The independent evaluator replay reproduces every report and the audit passes all
hash/config/flag checks (`results/p1_flatlands_independent_k128_validation_v2/audit.json`, 3 seeds,
0 failures). The compact portable hand-off is `site/data/flatlands_independent_k128_validation.json`;
per-seed prediction hashes are `b3a1e8d7…0958`, `b5c4f0d7…2030a`, and `35fa6906…5803` (full values
are in the snapshot and recovery ledger). Relative to the correlated K=128 candidate
(`0.09583 ± 0.01148` Brier), this is a matched spatial-correlation control only; both rows remain
validation-only and the non-official provenance/test-lock caveat is unchanged.

All three independent `best.pt` files were also restored with the current `PathRelNet` loader
(zero missing/unexpected state keys, `120,108` trainable parameters). The older `119,921`
parameter figure in the baseline table refers to the separate `MarginalCompletionBaseline`, not
to this matched independent decoder.

## Paired FlatLands K=128 effect-size check (completed 2026-09-03)

The two K=128 prediction matrices cover identical validation keys, so a paired cluster bootstrap
was computed rather than treating the 4,224 rows as independent. For each seed, complete
`(source_dataset, scene_id)` clusters were resampled 2,000 times and the delta was defined as
independent minus correlated (positive means worse for lower-is-better metrics). Brier deltas and
95% intervals are `+0.01011 [0.00286, 0.01839]`, `+0.02747 [0.01661, 0.03770]`, and
`+0.01673 [0.00816, 0.02585]` for seeds 20260831, 20260901, and 20260902 respectively;
the three-seed mean ± sample SD is `+0.01810 ± 0.00876`. The paired report also contains
NLL/ECE/false-safe and radius/source strata. This is an effect-size diagnostic, not a significance
claim or safety guarantee; the validation-only/non-official split and test lock remain in force.
The portable report is `site/data/flatlands_k128_paired_comparison.json`, generated by
`scripts/compare_flatlands_k128_paired.py`.

## Exact next actions

1. Preserve the completed FlatLands independent/correlated K=128 audits, compact snapshots, and
   paired report as validation-only artifacts; after any edit, regenerate only from passing audits
   and require exact replay, frozen hashes, 4,224 rows per seed, and zero radius violations.
2. The final regression gate has now passed: 80 tests, source/script compileall, `git diff --check`,
   26 JSON files/90 local site references, headless Chrome DOM smoke, full recovery SHA verification
   (344/344), FlatLands audits, and the UnScenes3D manifest/GPU/control checks (599/0/3).
3. Review the ground-robot panels at native resolution and keep the TUM desk asset only as a
   labelled geometry appendix; no target-derived panel may be presented as an input.
4. Keep the verified UnScenes3D control matrix and explicit replay/audit limitation in the paper
   hand-off. Do not pool unlike predictive objects or turn the coordinate control into a method
   claim.
5. Do not unlock `location_6` during this autonomous run. A separate explicit go/no-go review is
   required before any test labels are read; if it is approved later, unlock once and freeze all
   controls first.

## Recovery commands

```bash
cd /home/hairo/pathrel_transfer/pathrel_pro6000
git status --short --branch
PYTHONPATH=src .venv/bin/python scripts/verify_recovery_state.py --quick
# Full SHA verification (rehashes the 2.055 GB archive):
PYTHONPATH=src .venv/bin/python scripts/verify_recovery_state.py
sed -n '1,240p' CONTINUATION.md
sed -n '1,240p' results/p0_neural_cuda_contextplane_v4/report.json
sed -n '1,240p' results/p0_neural_cuda_contextplane_seed20260828_v4/report.json
sed -n '1,240p' results/p0_neural_cuda_contextplane_noreach_v4/report.json
sed -n '1,240p' results/p1_flatlands_query_audit_bounded/report.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/diagnose_p0_checkpoint.py \
  results/p0_neural_cuda_tuned01/checkpoint.pt --samples 128 --skip-events
./scripts/download_flatlands.sh --check
PYTHONPATH=src .venv/bin/python scripts/audit_flatlands_queries.py \
  --output-dir results/p1_flatlands_query_audit_bounded --overwrite
PYTHONPATH=src .venv/bin/python scripts/build_demo_site.py
```

`results/` and checkpoints are intentionally ignored. Any result used to make a research decision
must therefore be summarized here (and in the appropriate tracked research document) before a
session ends.

## Latest override: FlatLands valid-support correction and clean rerun (2026-09-03)

This section supersedes the earlier FlatLands model-number hand-off above. While building a real
checkpoint-derived website visual, the model path was audited against the frozen label geometry.
The label oracle treats `~epistemic_mask` as invalid/blocked, but the earlier neural forward left
those all-zero input cells stochastic. In a concrete validation query (`obs_156142`, candidate 3,
r=10), the fixed visual replay changed from ConPath/independent probabilities `0.727/0.797` without
the support clamp to `0/0` with it. The old K=32/K=128 map-derived numbers and qualitative panels are
therefore not eligible as current paper evidence. No FlatLands test label was read.

Implemented correction:

- `PathRelNet.forward` and `forward_features` accept `valid_support_mask` and clamp its complement
  to `BLOCKED` before posterior sampling;
- `collate_flatlands_replay` exports `valid_support_mask`, and the FlatLands trainer passes it in
  every train/validation forward;
- the trainer protocol is now `P1_BASELINE_PROTOCOL.md v1 + ConPath valid-support v2` and records
  the support policy plus implementation hashes;
- two model regression tests cover the clamp and mask-shape rejection, and a fixed-completion test
  proves probabilities outside explicit unknown support cannot create a path; the full suite passes
  **80/80**;
- `scripts/evaluate_flatlands_support_clamped.py` is validation-only and rejects test access. Its
  accelerated discrete-disk/four-neighbour implementation self-checks against the canonical NumPy
  clearance/merge-tree oracle before loading a checkpoint.

All six old K=128 checkpoints were then re-evaluated with K=128, the corrected support boundary,
the same three seeds, 4,224 validation rows per seed, and no test access:

| Decoder | Brier | NLL | ECE | False-safe @0.8 | Coverage @0.8 |
|---|---:|---:|---:|---:|---:|
| Correlated ConPath | **0.08467 ± 0.01350** | **0.40035 ± 0.06452** | **0.08979 ± 0.01627** | **0.03391 ± 0.00722** | 0.28150 ± 0.02290 |
| Independent | 0.10352 ± 0.00388 | 0.84715 ± 0.01366 | 0.09964 ± 0.01038 | 0.05468 ± 0.00678 | 0.33801 ± 0.00779 |

The independent-minus-correlated paired Brier delta is `+0.01886 ± 0.00963`; the three per-seed
2,000-draw scene-bootstrap intervals are `[0.00127, 0.01851]`, `[0.01888, 0.03833]`, and
`[0.00963, 0.02725]`. Thus the directional correlation advantage survives the correction, but this
is still a **post-hoc evaluation of checkpoints trained under the faulty forward**. Its ignored
artifacts are under `results/p1_flatlands_support_clamped_posthoc_k128_validation/`.

The homepage now uses
`site/assets/flatlands_k128_support_clamped_advantage.png` instead of the TUM desk video as its main
visual. It is rendered only from real checkpoints and the frozen validation replay by
`scripts/render_flatlands_k128_advantage.py`; the companion
`site/data/flatlands_k128_support_clamped_advantage.json` records prediction/checkpoint hashes,
fixed visual seeds, aggregate values, and label-derived case-selection rules. The TUM video was
moved to the geometry-pilot section. A 1440×1200 headless-Chrome screenshot confirmed that the hero
loads and is legible.

The execution-log paragraphs below are historical snapshots from the overnight run; the latest
2026-09-04 update at the end of this file is authoritative for process state. Clean formal training
completed in new directories and did not overwrite the old audit trail:

- correlated: `results/p1_flatlands_conpath_k128_support_clamped_v1/seed{seed}_conpath/`;
- independent: `results/p1_flatlands_independent_k128_support_clamped_v1/seed{seed}_independent/`;
- controller logs: `results/p1_flatlands_support_clamped_training_logs/`.

The three correlated seeds were launched concurrently first; the controller launches the three
independent seeds only after that group exits successfully. Each uses F=16, latent dimension 4,
eight train worlds, K=128 validation worlds in K=8 chunks, 40-epoch maximum, patience 8, the same
frozen queries/radii, and the corrected support mask. After completion, audit all six manifests,
rebuild the paired report/hero from the clean checkpoints, rerun site/recovery checks, and only then
decide whether the validation package is paper-candidate quality. Do not open the FlatLands physical
test split or UnScenes3D `location_6`.

### Live audit update (2026-09-03 22:57 EDT)

The three correlated clean runs completed two epochs without OOM. Epoch time is approximately
483 seconds per concurrent run. Validation Brier trajectories are `0.22797 -> 0.17264` (seed
20260831), `0.22749 -> 0.23764` (seed 20260901), and `0.23661 -> 0.24186` (seed 20260902); the latter
two have used one of eight patience steps. Atomic `best.pt`, `latest.pt`, and `progress.jsonl`
artifacts exist for every seed. These are early training values, not final comparisons.

The same boundary review was applied to UnScenes3D without reading `location_6`. Its exact oracle
uses `target_valid` as the support domain, while the older map-model forwards did not clamp the
complement before sampling. Therefore all historical UnScenes3D ConPath, independent-decoder,
mean-map, calibration-map, and qualitative posterior artifacts are now explicitly superseded for
paper/cross-domain claims; the coordinate-query and train-fitted radius-prior controls are not map
samplers and remain unaffected. `scripts/train_unscenes3d_conpath.py`,
`scripts/evaluate_unscenes3d_conpath_mean_map.py`, and
`scripts/render_unscenes3d_qualitative.py` now pass the validity mask into `PathRelNet` and record
the support contract. A synthetic forward smoke confirmed zero sampled safe cells outside support.
`UNSCENES3D_PROTOCOL.md`, the paper draft, README, and site now expose the reopened gate. A clean
UnScenes3D rerun must follow the FlatLands publication gate; do not promote the archived numbers.

`scripts/wait_and_finalize_flatlands_valid_support.sh` is also active as a fail-closed supervisor.
It waits for all six `run.json` files, aborts if trainers disappear for three consecutive checks,
then runs strict support/checkpoint/prediction/replay audits, builds compact snapshots and the clean
paired report, and invokes the figure renderer only if the aggregate Brier delta and all three
per-seed lower bootstrap bounds are positive. It writes a separate
`site/assets/flatlands_k128_clean_candidate.png`; it does not silently promote that candidate into
the homepage, so the final numbers/image still require visual and document review.

A validation-only UnScenes3D post-hoc diagnostic was also run on the three old correlated
ground-valid checkpoints with K=128 posterior means and `target_valid` clamped. It produced 4,587
event rows per seed, zero radius-order violations, and deterministic mean-map event Brier
`0.54740 ± 0.00251`, versus the archived unmasked value `0.62774 ± 0.00145`. Mean hidden-map Brier
is approximately `0.15017`, nearly unchanged, so the event shift is specifically attributable to
the invalid-support route. Outputs are under
`results/unscenes3d_support_clamped_posthoc_mean_map_k128/`; every report says
`posthoc_checkpoint_evaluation=true`, `retraining_required=true`, and `test_evaluated=false`.
This result prioritizes the clean second-domain rerun but is not paper evidence.

### Automated second-domain continuation (2026-09-03 23:39 EDT)

The clean FlatLands correlated runs have completed seven epochs. Current best validation Brier is
`0.06517` (seed 20260831, epoch 6), `0.10604` (seed 20260901, epoch 6), and `0.07624`
(seed 20260902, epoch 7). All three processes remain healthy; the last epoch improved seed
20260902 substantially while the other two used one patience step. These are live selection values,
not final paired results.

`scripts/wait_train_and_finalize_unscenes3d_valid_support.sh` ran behind the FlatLands
controller/finalizer and did not share its GPU lane. It completed the following steps after both
FlatLands groups exited:

1. train three correlated and three matched independent F=16 UnScenes3D adapters from scratch with
   the `target_valid` clamp on the 478/62 train/validation frames;
2. restore and audit all six selected checkpoints without loading frames or labels;
3. run K=128 support-consistent mean-map evaluation for all six checkpoints, replay all 4,587
   validation rows per seed, compare identical keys with whole-scene bootstrap intervals, and keep
   `location_6` locked;
4. render a clean qualitative candidate from the best correlated validation seed and copy only
   candidate-named artifacts into the site tree; homepage promotion remains manual; and
5. rerun the unit, compile, and diff-integrity gates.

The new reusable audit/comparison tools are `scripts/audit_unscenes3d_conpath_clean.py` and
`scripts/compare_unscenes3d_k128_paired.py`. The comparison independently reproduces all five
published event metrics exactly on the existing post-hoc K=128 CSV as a preflight check. Clean
outputs will use new versioned directories and will not overwrite the superseded historical runs.

A full PathRelNet call-site scan also found two legacy FlatLands utilities that predated the
amendment. `scripts/evaluate_flatlands_conpath_mean_map.py` and
`scripts/render_flatlands_qualitative.py` now pass `epistemic_mask` explicitly, distinguish clean
from post-hoc checkpoints in their JSON, reject test rendering before dataset/checkpoint I/O, and
fail early on invalid posterior chunk sizes. A one-scene validation-only CPU replay passed after
the correction; no test archive member was opened.

The live checkpoints are written by Python 3.13, whose concrete `Path` pickle class is
`pathlib._local.PosixPath`; Python 3.11 cannot deserialize that configuration object. Both
finalizers now use the same Python 3.13 environment as training. All three epoch-9 `latest.pt`
files restored successfully there with the valid-support v2 protocol and finite model tensors, and
the 80-test suite passed in that exact environment. The training processes were not restarted.

### Live audit update (2026-09-04 03:12 EDT)

The clean FlatLands correlated group has now completed all three exact validation runs. Each
directory contains a finite strict-restorable `best.pt`, a 4,224-row label-free prediction CSV,
an exact validation report, and `run.json` with the valid-support-v2 implementation hashes;
all three reports remain `paper_result=false` and `test_evaluated=false`. Scene-weighted Brier is
`0.05948` (seed 20260831), `0.06522` (20260901), and `0.07778` (20260902), mean
`0.06749 ± 0.00936`; this is still validation-only on the non-official provenance split. The
matched independent group was launched automatically at 03:05 EDT in its separate versioned
directory and is currently caching the same frozen train/validation packets. No test labels or
UnScenes3D `location_6` data have been opened.

### Live audit update (2026-09-04 07:28 EDT)

The clean FlatLands matched-independent group has now produced all three checkpoints and is in the
finalizer's report/audit hand-off. The clean paired package is complete: ConPath scene-weighted
Brier `0.06749 ± 0.00936`, independent `0.09521 ± 0.00703`, paired delta `+0.02772 ± 0.00993`,
and all three per-seed bootstrap intervals are positive. The checkpoint-derived visual is
`site/assets/flatlands_k128_clean_candidate.png`; it is explicitly validation-only/non-official and
does not open the physical test split.

The UnScenes3D supervisor has released the FlatLands GPU lane. Its three correlated and three matched
independent `target_valid`-clamped adapters completed in their separate versioned directories. The
supervisor then ran strict checkpoint audits, K=128 mean-map replay, a paired comparison, and
candidate-only qualitative rendering. `location_6` remains locked; no test artifact was loaded.

### Live audit update (2026-09-04 07:50 EDT)

The UnScenes3D clean-support supervisor has now completed all six F=16 validation runs. The
correlated and matched-independent checkpoints restored strictly, passed finite-tensor and
support-policy audits, and reproduced the deterministic K=128 mean-map comparison byte-for-byte.
The clean event Brier is `0.54704 ± 0.00218` for correlated ConPath and `0.54740 ± 0.00251`
for the independent decoder (paired delta `+0.00036 ± 0.00062`), so this two-scene adapter is a
transfer/support diagnostic with no measurable correlation advantage. Real-checkpoint positive and
false-safe panels are published as validation diagnostics in `site/assets/`; `location_6`, test
labels, and all official test artifacts remain locked. Final work is documentation/site/recovery
consistency review only; no training process is active.

### Final hand-off update (2026-09-04 08:21 EDT)

The clean UnScenes3D table was widened for desktop readability and its artifact links now wrap
without stray separators. The final regression pass used the Python 3.13 checkpoint environment:
80 unit tests passed, source/scripts compile and `git diff --check` passed, the static site audit
found 26 JSON files, 90 local references, 20 images, no missing alt text, no duplicate IDs, and no
missing local targets. Both clean FlatLands audits and both clean UnScenes3D audits report three
seeds/zero failures; the paired reports reproduce byte-for-byte. The UnScenes3D control audit is
599 checks/0 failures/3 intentional legacy warnings, and the full recovery verifier is 344/344
with SHA-256 enabled. The refreshed published control-audit copy matches its source. No training
process is active and no locked test artifact was opened.

## Plan reconciliation and website publication preparation (2026-09-06)

The runtime goal lookup returned no active goal. The seven-stage roadmap remains a research
objective, while the September 2 overnight plan is historical and its bounded audits are complete.
`WORK_PLAN.md` now separates version/site publication, clean analysis, missing clean ablations,
second-domain diagnosis, scalable-operator evidence, and the still-locked final test decision.
Recovery commands now use the Python 3.13.13 checkpoint interpreter and explicit clean-support
roots instead of the superseded directories and the local Python 3.11.15 venv.

The three real-checkpoint PNGs were visually inspected and are linked from the site, including
full-resolution UnScenes3D positive and false-safe panels. The page's method description now
describes the learned BEV model; old calibration traces are explicitly historical, stale ConPath
comparisons were corrected, and the 0.8 false-safe claim now also discloses unequal coverage.
A mobile overflow caused by the reproduction code card was fixed, and the narrow navigation wraps.

The Python 3.13 suite passes all 80 tests; all 61 Python source/script files parse, both three-seed
FlatLands clean audits pass, and both three-seed UnScenes3D clean audits pass. Chrome verified
1440- and 390-pixel layouts with zero broken images, no page overflow, current model numbers and
four dynamically rendered fixed-baseline rows. The final site audit and Git publication are
recorded under `results/maintenance_20260906/`. The earlier 599-check control audit remains an
archived structural audit of the historical control package, not new scientific evidence.
No new training or locked test read was performed for this publication step.

## 2026-09-06 paper evidence and final-projection correction

Git publication `2deced0` / `0000508` completed first; Pages run `34077832249` succeeded.
Paper analysis source is committed in `392017b`, and the final mean-map projection/bound audit
in `a00c86a`. The shuffle helper recorded in the frozen reports is recoverable exactly from
`392017b:src/pathrel/posterior_audits.py`; the later module adds an independent projection helper.

New evidence roots:

- `results/paper_clean_checkpoint_controls_v1`: six exact RNG replays, nested K=32/64/128,
  deterministic mean maps and hidden-map metrics; original K=128 event drift is zero.
- `results/paper_clean_marginal_shuffle_v1`: three controls preserving every empirical cell count;
  Brier becomes 0.16139 ± 0.00677, with positive paired Brier intervals in every seed.
- `results/paper_clean_analysis_v1`: nine methods, 23 prediction files, identical 4,224 events,
  142 contributing scenes, whole-scene paired bootstrap and fractional label-blind boundary ties.
- `results/unscenes3d_ground_valid_support_clamped_mean_map_k128_v2` and its `independent_`
  companion: unchanged six clean checkpoints with the final mean-map support projection fixed.
- `results/unscenes3d_ground_valid_support_clamped_k128_comparison_v2`: corrected paired replay.
- `results/unscenes3d_ground_valid_support_clamped_qualitative_v2/seed20260901`: newly rendered
  positive and radius-2 false-safe panels. Label-based diagnostic selection is disclosed.
- `results/unscenes3d_observation_ceiling_v1`: exact pessimistic/optimistic bounds over all
  51,288 train/validation events, plus 27,522 corrected model predictions checked against them.

The mean-map bug was found by the independent bound audit: v1 restored observed-free cells
outside target_valid after correct model sampling. The shared projection now gives blocked
observations precedence and applies support last. Training does not need to be repeated for
this correction. Old v1 reports/images remain in their original results directories and Git
history; their event numbers must not be used as current support-consistent evidence.

The working manuscript is rewritten around the current evidence; its earlier version is
`PAPER_DRAFT_HISTORY.md`. Five SVG/PDF figure pairs and compact JSON are included in the site.
Validation: 89 unit tests, exact numeric/hash replay, 79 local HTML links, 29 JSON snapshots,
25 image elements, and Chrome 1440/390 px checks pass. Full recovery verification passed all 438 entries with SHA-256 enabled; hashes are recorded in
RECOVERY_STATE.json. Neither physical test was evaluated.

No active training or runtime goal is registered. Next: clean three-seed no-event/no-global
training under the fixed FlatLands protocol; the shuffle intervention does not replace it.
The deterministic-control gap, unstable equal-coverage advantage, small second-domain sample,
and absent scalable backward operator remain explicit final-paper limitations.

## 2026-09-07 clean ablation matrix launched

The user requested the next step. Implementation `e6f0910` adds the frozen six-run
matrix, configuration guards, safe resume/finalization, per-run strict audit and
paired analysis. Trainer/model/data hashes exactly match the original clean full
model. The full model is reused; no_event changes only reachability_weight to zero,
and no_global changes only disable_global_factors to true. The protocol and analysis
rules are in CLEAN_ABLATION_PLAN.md. All 93 unit tests pass, and the new audit replayed
the original seed 20260831 full-model Brier exactly (0.059479796640678484).

Supervisor PID at launch: 4078565. Initial no_event worker PIDs: 4078837 / 4078838 /
4078839; the three no_global runs are queued. These are launch identifiers, not a
promise that the same PID is still alive. Read the current ignored progress file:

`results/paper_clean_ablation_matrix_v1/progress.json`

The detached supervisor continues independently of this conversation, at most three
workers at a time. It preserves the original training code and exact evaluation.
The immutable matrix records complete commands, reference report/checkpoint hashes,
training source hashes and launch revision. Do not launch duplicate workers. If the
supervisor stops, inspect supervisor.log and per-worker logs before using the same
runner command to resume; ambiguous directories are deliberately refused.

After all six runs finish, the supervisor automatically audits them and writes the
paired JSON, manuscript table and SVG/PDF candidates under `analysis/`. Review those
results before incorporating them into PAPER_DRAFT.md and the published site. Both
physical test sets remain locked. No runtime goal was registered.

## 2026-09-07 user-requested training pause

At 05:50 UTC / 01:50 EDT, the user requested a pause after the runtime estimate.
SIGTERM to the verified supervisor triggered its existing graceful SIGINT cleanup
for all three workers; the queue stopped and all four recorded processes exited.
The three no_event runs retain complete epoch-4 latest checkpoints and their best
checkpoints; none of the no_global runs has started. No final ablation result exists.

`results/paper_clean_ablation_matrix_v1/pause_checkpoint_audit.json` verifies strict
model and optimizer restoration, finite tensors, the frozen configuration and RNG
states for all nine latest/best/interrupted checkpoint files. When the user requests
resumption, use the existing runner and its `latest.pt` recovery to start epoch 5.
The incomplete fifth epoch will be rerun. `interrupted.pt` includes partial-epoch
state and must not replace the complete-epoch `latest.pt`. Do not resume merely
because an older continuation section describes an active supervisor.

## 2026-09-07 Chinese website redesign

The user asked for a simpler website, understandable figure legends, better plots,
more dataset pictures and an explanation of dataset/baseline choices. They supplied
https://s-team-git.github.io/Lightweight-3DGS/ and its GitHub repository as the visual
reference and requested Chinese wherever possible. The public HTML/CSS and repository
file tree were inspected; the new implementation is framework-free and does not
require external fonts or scripts.

The home page now has four reading sections, two switchable checkpoint examples,
labelled S/G endpoint markers without connecting lines, five Chinese chart tabs with
separate mobile SVG/PDF exports, a folded nine-method numeric table, six FlatLands
training examples, six UnScenes3D training scenes and an 18-frame synchronized camera /
observation / reference browser. Twenty-three unique camera images preserve their
released bytes. No training resumed; the four model/case visualization replays
are the previously published validation cases. Their original event probabilities
are retained, and the separate fixed visualization RNG is disclosed.

The original white/yellow straight lines were query connectors, not planned paths.
TUM's black lines were camera MoCap trajectory and its colored segments were changing
query pairs. Old media remains archived with a visible historical-page banner. New
maps use a consistent legend and distinguish unknown cells from invalid support.
The no-path example explicitly retains the high-probability failure. New dataset
examples use train-only scene IDs / timestamps, not outcome-based selection.

Official sources confirm PaSCo and SGN use SemanticKITTI / SSCBench-KITTI360; S4C uses
KITTI-360, and the online SceneSense work uses its own real-building occupancy data.
The site and DATASET_CHOICE_ZH.md distinguish those original systems from our adapters.
UnScenes3D's obsolete baseline-summary value was corrected to the existing v2 result.
Main numerical evidence and the training/model/data implementations are unchanged.

Build `scripts/build_site_visuals.py` then `scripts/build_site_page.py`; the latter is
also required after the older English paper-table generator. Read-only audits are
`scripts/audit_site_visuals.py` and `scripts/audit_paper_evidence.py --output
results/site_redesign_20260907/paper_evidence_audit.json`. The output option preserves
historical audit receipts. `scripts/check_site_browser.mjs` exercises real Chrome at
1440/390 pixels, including all gallery/plot switches, image modal and Escape, frame
seek/play/pause and mobile step navigation. Receipts/screenshots are under
`results/site_redesign_20260907/`; deployment status is recorded there after push.
This historical pause was superseded by the explicit resume request recorded below.

## 2026-09-07 evening: approved experiment plan and training resumption

The user explicitly approved continuing training and carrying the proposed comparisons into the paper.
The six-run clean ablation supervisor resumed at 21:42 EDT from complete epoch-4 checkpoints;
all three no_event workers subsequently completed epoch 5. Full training configuration and history
are preserved. Inspect live progress for newer epochs; no_global follows automatically in the queue.
The top recovery snapshot supersedes historical pause instructions throughout this log.

`EXPERIMENT_DESIGN_ZH.md` now has the confirmed paper deliverable checklist. Required external
comparisons must remain in future work, alongside statistical, cost, ablation and failure analyses.
`EXTERNAL_DATA_AUDIT_ZH.md` records the first train-only CogniPlan audit: 3,000 full and 23,795 partial
maps, no observed/target conflicts, distinct pixel-identical target hashes, and 3,000 same-name but
different-content IDs across predictor/planner train assets. Namespace those IDs by asset; do not
infer overlap from filenames. Released checkpoint batch size is 24, whereas repository default is 32.
No external model inference or final test ran. Preserve official pixel conversion and distinguish
public-checkpoint train sanity checks from validation of newly retrained models.

## 2026-09-08 完成六组消融评估与中文汇总

最新工作（2026-09-08）：用户要求完成评估汇总并继续收尾。本轮六组消融已于
08:56:19 UTC（04:56 EDT）完成精确评估、冻结审计与配对统计；调度器3081128和全部worker已退出。
不要重复启动这套已完成矩阵。当前任务是完成中文汇总、论文、网站与Git发布，然后进入已确认的外部实验接口/尺度/测速阶段。

三种模型的三次训练 Brier：完整0.06749±0.00936、no_event 0.20425±0.00322、
no_global 0.09499±0.00157。六个逐种子Brier区间均为正，支持完整模型；
六个等30%覆盖率风险区间均包含零，不支持稳定安全收益。no_global的风险点估计在两个种子反向。
no_event停在12/9/17轮，选择4/1/9；no_global停在12/18/18轮，选择4/10/10。
半径20格的no_event全部预测为0，但场景等权有路率20.67%；该失败诊断不能扩大尚未核对的物理尺度主张。

权威统计：`results/paper_clean_ablation_matrix_v1/analysis/report.json`。
中文报告：`EVALUATION_SUMMARY_ZH.md`；论文新增5.5节；网站独立三模型表和中文区间图。
两个此前固定的验证例子各三模型，输出6幅概率图和6幅首个真实采样按机器人尺寸收缩后的图，未按新结果挑例或挑样本。
图片采用CPU新绘图随机流；标题事件概率来自原CUDA精确验证CSV，随机流不同，不能要求两者精确相等。
有效绘图记录：`results/ablation_publication_20260908/cases_v2/report.json`（初版概率图记录在cases/，保留但不作当前来源）。
像素12/12、首个世界连通性6/6、9个预测CSV主指标/分层/配对Brier重算通过；98项测试通过。
最后桌面1440/手机390交互检查通过：`results/ablation_publication_20260908/browser_final/report.json`。
本轮独立数值/网页/版本/部署收据统一在 `results/ablation_publication_20260908/`；以完成收据判断是否已推送与线上核验。

已确认实验方案长期有效，见 `EXPERIMENT_DESIGN_ZH.md` 第8节；只将当前有界验证消融标记完成。
必做主比较是FlatLands原生LaMa/4成员集成、FM+XAttn和CogniPlan原生地图生成模块，
并保留强简单对照、三次独立重复、成本曲线、场景统计和失败分析。外部方法尚未运行。
先做数据/接口、原生质量、固定训练样本测速与具体配置冻结；扩大数据时全部相关方法另开一致的新版本。
正式测试仍锁定；不能因续训授权跳过这些科学门槛。没有注册runtime goal。

### 保留的上一轮恢复快照（历史，已由完成状态取代）

最新指令（2026-09-07 23:59 EDT）：用户再次明确要求继续训练，已解除23:47的暂停。
调度器与三个精确评估进程已经启动；先检查现有进程，不能重复启动。论文实验方案与冻结协议保持不变。
先读 `EXPERIMENT_DESIGN_ZH.md` 的完整方案与第8节论文交付清单，再读 `WORK_PLAN.md`。
必做主对比为 FlatLands 的 LaMa/4成员集成、FM+XAttn，以及 CogniPlan 原生地图生成模块；
保留强简单方法、三次独立重复、成本曲线、场景配对统计和失败分析。KITTI-360 仍为条件扩展。
外部实验先完成数据/接口、原生质量、测速和具体配置冻结；不能因恢复旧消融跳过这些步骤。

Training resumed again at 23:59 EDT (2026-09-08 03:59 UTC), following the 23:47 pause.
The three no_event seeds completed 12/9/17 epochs; all reached the frozen patience=8 stop.
Their best epochs are 4/1/9. All six latest/best checkpoints passed model/optimizer restore,
finite-value and RNG-presence checks. The three no_global runs have not started.
Pause receipts and checkpoint backups: `results/training_pause_20260908T034736Z/`.
Current resumption receipts: `results/training_resume_20260908_finalization/`.
The launcher is performing final evaluation only for no_event, without an extra training epoch.
Interrupted exact evaluation saved no prediction rows and has restarted from the beginning with
best weights and the saved latest RNG state. Do not substitute old interrupted.pt.
The existing supervisor automatically starts queued no_global workers after a completed run passes its audit.
Use `results/paper_clean_ablation_matrix_v1/progress.json` and each command's --finalize-dir flag
to distinguish the current evaluation phase from parameter training; epoch counts do not advance in evaluation.
Earlier authorization and the accepted-design snapshot remain in `results/training_resume_20260907/`.
