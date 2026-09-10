# 正式外部矩阵：执行与收尾

> **2026-09-10 论文收敛指令（覆盖下方历史计划）：** 外部模型训练/评估和自动续训已取消，保留全部结果；不扩数据、不新建模型方向、不再调参、不打开最终测试。已有原ConPath父级隔离三seed触发停止条件，当前仅整理论文及可审计图表。LaMa/FM只保留Related Work与欠收敛补充记录。当前计划见 [WORK_PLAN.md](WORK_PLAN.md)，证据见 [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md)。

这是当前冻结协议的操作说明，不修改协议、训练实现、评估口径或测试锁。正式实验仍在进行，本文件不是结果通过凭据。

## 当前入口

- 协议：`results/flatlands_external_formal_protocol_v1/protocol.json`，SHA256 为 `409c8bc4a9142d75a62e6c3d67204ef7d5045ad38f74b45b5411f8bfe899c0e2`。
- 当前队列：`results/flatlands_external_formal_v1/queue.json`。
- 阶段状态：`results/flatlands_external_formal_v1/stage_status/{lama,flow}.json`。
- 训练：`results/flatlands_external_formal_v1/runs/<method>/<outer_seed>/member_<n>/`。
- 完成的阶段：`results/flatlands_external_formal_v1/stages/<method>/<outer_seed>/<stage>/attempt_<n>/complete.json`。

读取状态时以进程、最新日志和完成回执为准。状态文件中的目标步数不是已经完成的步数。LaMa 的成员编号从 0 开始，四个成员组成一次外层随机种子重复；四个成员不等于四次论文重复。

只允许两个并发 GPU 子任务，所有正式队列共用 `.queue.lock`。当阶段队列仍运行时，不另起完整训练队列。显存 60 GiB 是上限预算，不需要为了占满显存提高 batch。

## 阶段结束后的检查

1. 先检查每阶段完整校准与验证回执、固定输入和预测文件 hash，再用 `scripts/audit_flatlands_formal_stage.py` 做独立 CPU 重算。审计只读已保存世界地图，不再推理，不访问测试图。
2. 使用 `scripts/render_flatlands_formal_convergence.py` 展示固定案例在未训练、1000 步、5000 步的全部世界地图；人工逐图核对全空、全障碍、碎块化、覆盖证据等退化情况。渲染脚本不能自动宣布视觉检查通过。
3. `scripts/seal_flatlands_formal_full_plan.py --method <method>` 生成数值门槛回执。数值未通过时保留失败结果，不开始完整训练。
4. 只有数值与实际图片检查同时通过，才把包含正确协议、方法、种子、固定案例 hash 和三个阶段身份的视觉回执传给同一封存脚本的 `--visual-review`。脚本按已经冻结的规则生成 10000 或 20000 步计划，不能人工挑更有利的步数。

每次审计或作图使用新的结果子目录。没有 `complete.json` 的中断评估不能计作完成；旧的部分预测和日志保留。训练可显式恢复，已经成功完成的评估应验证后复用。

## 完整训练与三次重复

完整入口为 `scripts/run_flatlands_formal_full.py --methods lama flow`。这是需要有效封存计划的入口，不是看到本文件就可启动的命令。首次启动与恢复区分；已有合法流水线必须显式 `--resume`，未经登记的非空目录拒绝使用。

三个外层种子固定为 `20260831 / 20260901 / 20260902`。每个种子所有 LaMa 成员达到共同检查点后才能计算 K=4 校准指标。首种子的 smoke/pilot 已有评估直接复用；完整阶段其他检查点只做校准，选定后才做一次成功的验证评估。

选择规则覆盖注册的全部候选：校准集原始 Event Brier 最小；与最小值相差不超过 1e-5 时比较地图 vote NLL，再选较早步数。不同方法、种子允许选中不同步数，报告按完成完整训练后选中的模型汇总，不能把不同步数拆成不完整的三种子矩阵。

如果最优模型仍来自短训练，保留这个选择，并标记为诊断。不能用较差的长训练权重替换它来获得主表资格。完整预算结束但校准指标仍明显变化，也不能称为充分收敛。实际总训练成本包含全部成员、未选中的后续训练以及中断重放；缺失进程结束记录时耗时只能标为下界。

当前协议也登记了 ConPath、Independent completion、原 ConPath、去可达性损失、确定性补全和直接查询控制。完整比较必须在本轮冻结的新数据和查询上完成这些方法的三次重复。旧数据协议上的 K=128、短训练 profile 或旧控制审计不能填入这次矩阵；它们也不需要重跑。

## 最终审计与论文资格

逐种子保存 `full_runs/<method>/seeds/<seed>/selection.json`、`complete.json` 和版本化收敛回执。最终图片检查须绑定选中校准结果和末三个校准检查点的 hash；补入视觉检查后恢复调度，只更新新的收敛回执，不重复已完成的验证推理。

最终独立审计必须把分开的选中校准目录与验证目录绑定起来，重新核对：三种子和 LaMa 四成员、实际训练完成度、校准选点、全部固定查询、地图和事件分层汇总、仅校准集拟合的映射、输入证据与支持范围、成本，以及测试锁。阶段审计通过不能代替最终审计通过。

只有真实三种子完整产物齐全后，运行 `PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python scripts/audit_flatlands_formal_final.py --method lama`（FM使用`flow`）。脚本只读已保存的校准/验证NPZ与checkpoint，不重新推理，不读取原始测试图；每次保留独立attempt，全部通过才新建`final_audits/<method>.json`并绑定实际verification的SHA。它还独立重算全部校准候选以核验选点，检查恢复RNG终态、Adam状态及实际训练耗时。不得手工填写最终passed来补缺项。

pilot视觉回执必须指向真实收敛图manifest，绑定三个阶段metrics SHA和每例预测SHA，包含三个阶段×18项固定案例的54条实际审查记录；其中多半径项虽复用一般案例，也按类别单独登记。必须明确pilot质量与持续退化检查结论；未训练阶段已知的随机/全空现象可以如实解释，不能据此跳过pilot检查。精确字段见最终审计脚本模块说明。网页排版审查只证明图能看清且数值来源正确，不授予pilot或论文资格。

`summary.csv` 与 `metrics.json` 保留缺失值、实际 K、样本数、半径、原始事件分数和映射后诊断的区别。图片不按模型成绩挑选；固定案例中的失败例和所有世界地图都保留。单一方法完成不等于 ConPath 已经优于它，也不等于整个比较矩阵已完成。

最终 `report.md` 根据充分训练后的真实结果回答六个问题：未知地图能否补全，地图改善是否转化为事件校准，ConPath 相对普通补全的效果，空间相关性和 reachability loss 的分别作用，主表与诊断的边界，以及是否有依据进入真实室内数据。若没有足够证据，写清未完成或假设不成立；不提前扩展方向。

## 暂停与故障

用户要求暂停时，先向当前正式队列主进程发送一次 SIGINT，并等待子任务退出和恢复文件落盘；不要停止其他项目进程。暂停标记在用户要求继续之前保持有效。重新启动前核对协议、源码 hash、队列锁和 checkpoint 配方。

失败记录包括原因、最后有效 checkpoint、原始 loss 日志、峰值显存、实际耗时及下界状态。区分实现错误、资源不足和尚不能归因的方法失败。没有充分训练的外部方法不能用来支持正式优越性结论。
