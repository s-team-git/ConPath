# 论文主张与最终持出资格审稿核对

2026-09-10 UTC。作者结论：**现有资料足以写一篇结论克制、来源明确的完整工作稿，但没有证明当前六方法矩阵完整，也没有证明最终持出集合合格。** 当前应保留原版 ConPath，整理已有证据，不因证据缺口自动启动训练或打开测试。

本次只读已有论文、冻结数字快照、划分与访问清单；没有读任何原始地图、保存世界、模型 checkpoint、最终测试资产或 `location_6` 内容，没有运行训练、推理、几何重评估，也没有重复上一轮全部 189 个来源文件审计。机器记录见 [主张审计](results/paper_submission_v1/review/claim_audit.json)、[数值核对](results/paper_submission_v1/review/numeric_claim_checks.json)、[持出资格审计](results/paper_submission_v1/review/holdout_eligibility.json) 与 [实际输入文件及 SHA](results/paper_submission_v1/review/read_manifest.json)。这些记录绑定审核时的文件 SHA，不代表后续编辑自动得到同一审核。

## 1. 逐个研究问题：能写什么，不能写什么

| 问题 | 同协议证据 | 必须保留的反例及缺口 | 结论强度 |
|---|---|---|---|
| RQ1：相同边际下的空间相关性是否有用？ | 历史 K128 有精确保持每格经验自由计数的世界索引置换；Brier 从 0.06749 变成 0.16139。当前独立训练比较中，ConPath K32 为 0.10380，independent 为 0.11201，三个种子均改善。 | 这两类证据属于不同队列。当前连续 hidden-map Brier 0.15147 与 0.15158 接近，不等于分布相同或通过等价性检验。历史均值图 Brier 0.06957 接近 ConPath，并在一个种子反超；当前 deterministic 的 Brier 差值区间跨零。 | 历史有限样本机制成立；当前开发比较支持 independent 对照，不能拼成一次未污染、严格固定边际的泛化实验。 |
| RQ2：事件监督和全局因子是否改善事件评分？ | 历史 full / no-event / no-global Brier 为 0.06749 / 0.20425 / 0.09499，NLL 和 ECE 也支持 full；三个种子、相同历史 K128 查询与评估器。 | 当前父地点隔离队列缺 direct-query、no-event、no-global。历史 no-event 的 0.8 风险反而更低，但覆盖也更低；相同 30% 覆盖的风险区间跨零。no-event 仍用 Event Brier 选 checkpoint；no-global 只删除解码器低秩因子，未删除全部上下文或空间依赖。 | 只支持历史设置中的训练机制诊断，不能宣称当前消融闭合，也不能宣称所有校准和安全指标全面改善。 |
| RQ3：半径、遮挡、窄瓶颈和不可达时是否可靠？ | 相同世界加精确嵌套圆盘几何保证可达概率随半径不增；已有半径、来源、真值分层、reliability 与固定案例。 | 半径 20 格时，ConPath 三种子在置信度 0.8 的覆盖均为零，风险未定义；该层 Brier 0.06142 劣于 deterministic 0.04025。半径 10 格的风险/覆盖均值为 0.30851/0.03547。严格内部窄瓶颈及预先注册的遮挡分层扫描未提供。 | 单调性是结构性质；各半径均可靠、可安全执行或窄瓶颈泛化仍未证明。 |

可直接使用的英文措辞：

**RQ1:** “A fixed-marginal intervention on an archived development cohort shows that the alignment of sampled worlds contains useful path-event information beyond their empirical cell marginals. A separate parent-isolated development comparison favors ConPath over independently sampled completion.”

**RQ2:** “Within the archived historical protocol, event supervision and low-rank decoder factors improve Event Brier, NLL, and ECE. The overlap-affected cohort and incomplete current ablation matrix limit causal and generalization claims; uniformly improved selective risk is not established.”

**RQ3:** “Shared-world footprint evaluation guarantees monotonic reachability forecasts as the disk radius increases. Existing development diagnostics assess reliability across recorded radii and query classes, but do not establish uniform calibration in occluded or narrow-bottleneck environments.”

## 2. 数字一致，不代表所有指标占优

针对支撑正文主张的 13 行，重新用保存的逐种子数值核对了 **78 项均值与样本标准差**；另核对正文 **19 个关键显示数字**与原值四舍五入一致，并从快照的已有半径汇总核对上述 6 个方法—半径组合。均通过。没有从原始预测或地图重新计算指标。

需要让读者同时看到：

- 当前 ConPath 相对 independent 的 Event Brier 差值为 **0.0082119**，记录的 95% 配对父地点区间 **[0.0043906, 0.0125084]**。这支持当前开发样本及三个指定种子的比较。相对 deterministic 的区间 **[−0.0161614, 0.0337657]** 跨零。
- 当前 deterministic 的**连续概率地图** hidden-map Brier **0.138685**，优于 ConPath 的 **0.151468**。不能拿 deterministic 阈值后二值地图的较差分数替换其连续概率分数。
- 当前训练集拟合的半径先验 NLL/ECE **0.484988 / 0.056531** 优于 ConPath **0.793301 / 0.086223**，但该先验的 0.8 覆盖为零，风险没有定义。不能写成“零风险模型”，也不能因此抹去它较好的 NLL/ECE。
- 历史 direct-query 的 ECE **0.040760** 低于历史 ConPath 的三种子均值，但它只有一个种子。另一种三种子 coordinate-query 控制不能代替它。
- 历史 no-event 的 0.8 false-safe **0.025608** 低于 full 的 **0.045525**，相应覆盖为 **0.193163 / 0.336443**。固定阈值的不同覆盖比较不能自动解释为安全性优势。
- 当前 **750/1545 个事件**已由观测约束确定不可达，ConPath 在这部分零错误；可由隐藏补全改变的子集 Brier **0.19912**，远高于整体 **0.10380**。子集采用自己的父地点归一化，不能用原始事件个数把两个平均数简单线性混合。

**Brier 下降是概率事件预测误差改善，不单独证明校准更好。** 校准、分辨能力、任务难度、有限样本极端概率都会影响分数。现有十箱 ECE 与 reliability 图补充了描述，但不提供按每种半径、来源或遮挡条件均校准的保证。

## 3. 不确定性的单位必须保持原样

| 数量 | 正确解释 | 禁止的解释 |
|---|---|---|
| 表内 mean ± SD | 真实训练种子的均值及样本标准差，分母使用 n−1 | 95% CI、标准误、1545 次独立实验的波动 |
| 当前父地点 bootstrap | 40 个父地点，在五来源各 8 地点内分层；条件于所显示的三种子平均 | 任意未来训练种子的置信保证或未触碰测试上的显著性 |
| 历史 bootstrap | 142 个贡献事件的子场景，逐种子重采样 | 修复了 27/160 验证观测父地点重叠的独立建筑区间 |
| 当前样本量 | 40 父地点、515 起终点查询、1545 半径条件事件 | 1545 独立场景；K32 世界等于 32 次训练重复 |
| 两种子候选 | 仅 20260910/11 与原模型相同两种子的比较 | 候选二种子均值对原模型三种子均值 |
| 半径 | 0、10、20 grid cells | 未经独立尺度验证的米 |
| false-safe @0.8 | 在预测至少 0.8 的接受集合里，不可达事件的比例 | 所有负例上的假阳性率、覆盖率，或等 30% 覆盖风险 |
| 风险与覆盖均值 | 每种子先按等父地点权重形成比例，再平均种子 | 把所有种子事件混在一起得到同一个总体比例 |

NLL/ECE/0.8 汇总是保存验证预测的事后描述。不能改写为原父地点协议预注册的模型选择指标，也没有补齐当时未保存的逐查询 calibration 预测。图表的旧有评估和主张审核不是新的模型推理。

## 4. 最终 holdout：目前只能证明未获资格，不能证明不存在

**已有元数据不能证明一个完整、可立即解锁的最终持出集合合格。** 审计字段为 `eligible_holdout_proven=false`、`eligible_holdout_count=null`；没有建立或选择新测试集合。`null` 表示尚未证明，不能写成“可用测试为零”或“剩余所有数据均可用”。

能够确定的历史事实：

1. 更正清单列出 **53 个**曾读取图像的 physical-test 观测；旧训练中的 **8 个**与旧验证中的 **5 个**都是这 53 个的子集。不能相加为 66。历史亦记录 ScanNet++ 的 **32 个**已查看场景；本处不将其与 53 相加，也不声称两项互斥。不能将该来源整体称为 untouched。
2. 既有父地点审计显示，五个主要室内来源的 physical-test 父地点均也出现在 physical-train **档案**中。这是档案身份重叠，**不等于这些地点全部实际用于训练**。它说明目录名 `test/` 本身不能证明父地点独立。
3. 当前 100/25/40 的三份选中清单均来自 physical-train，内部父地点交集为零；但它们已经被用于训练、选模和开发验证，不能重新命名为最终测试。
4. 已取消外部实验的初始准备日志记录 **2292 个观测、1292 个父地点**的访问。质量隔离后的 eligible 清单只有 **2258 个观测、1273 个父地点**；被移出的 **34 个观测、19 个父地点**仍然已经检查过。暂停、取消训练或隔离样本不会恢复未查看状态。
5. 旧 320 观测、当前 165 观测、外部初始访问三份清单的父地点并集包含 **1532 个已知开发接触父地点**，已保存为 [部分开发接触清单](results/paper_submission_v1/review/known_development_exposure_parents.csv)。它只汇总三项已知历史，**不是完整排除证书或新测试清单**；没有出现在表里不代表从未被访问。

尚缺的持出资格元数据：

- 跨全部历史运行、人工预览、缓存目标、网页展示与会话的访问记录覆盖证明；更正文件和一个新日志只能证明已发生的访问，不能证明没有其他访问。
- 每个访问观测及任何未来候选观测的统一父地点身份。旧 512 观测查询审计表没有 `parent_group` 列，仅有旧 320 的父地点表不足以闭合完整访问史。
- 跨来源的同一物理地点/重复身份排除规则及未解析身份的隔离记录。当前三份开发集合的 D4 精确重复检查不能自动扩展为全档案去重证明。
- 对所有训练、calibration、validation、开发决策、质量审计访问做父地点层面的排除闭包。已被质量隔离的观测仍必须算接触历史。
- 后续独立决定的冻结评估清单、模型/种子/checkpoint、K、查询、阈值、权重和 evaluator 身份。此写作任务没有赋予最终测试权限，也没有创建这一清单。

旧 `test_evaluated=false`、清单中缺少某个 ID、以及本轮没有读取测试，均不足以单独建立完整 untouched 证明。当前继续锁定测试。

## 5. 作者现在可以推进的工作

把当前三种子结果作为明确标注 development-validation 的主要表；把历史固定边际、历史消融、两种子候选分别呈现，不填补新队列的缺失行。正文保留上述半径、均值图、确定性地图、先验和风险反例，让标题、摘要、结论与证据范围一致。

现有工作稿可以进入英文精炼、LaTeX 排版、图表引用核对和作者审阅。完整投稿证据仍有缺口；此判断不会自动触发补训、换数据集、重新选查询、拟合阈值或最终测试。LaMa/FM 继续仅作 Related Work 与未充分收敛的补充记录。
