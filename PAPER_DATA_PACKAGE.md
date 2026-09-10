# ConPath 论文数据封存与使用说明

更新：2026-09-10。当前已有实验数据已整理为可脱离原工作目录核验的[离线证据包](results/paper_submission_v1/evidence/README.md)。它保存原样报告和分数，并加入来自既有开发验证缓存的标量标签；**没有新训练、模型推理、最终测试读取或指标调参**。

## 交付内容

| 内容 | 数量 / 范围 |
|---|---|
| 原样归档的小文件 | 140 份，共 5,793,605 bytes；原论文来源 JSON/CSV、协议、训练选择记录、访问更正、实现源码文本和本轮标量导出 |
| 完整包 | 156 个文件，共 8,473,916 bytes（约 8.08 MiB）；含说明、库存、索引、脚本和本次核验回执 |
| 冻结总表 | 原 29 行聚合；原 snapshot JSON SHA 不变 |
| 逐事件分数 | 23 份固定方法 / seed / K CSV，每份 1545 个事件；40 个父地点、515 个查询、radius 0/10/20 grid cell |
| 开发验证标签 | 1545 行固定标量标签，与上述分数主键一一对应 |
| 分层视图 | 276 行逐 seed / 分层记录，36 行主方法三种子分层汇总；展开自已有 snapshot |
| 精确数字入口 | 24,272 条未舍入 JSON 数值路径，29 行 cohort/method/K/seed 索引 |
| 未复制的注册资产 | 51 个 NPZ 和 14 个权重文件，仅保留既有 SHA 与排除原因；无地图/权重字节进入包 |

这里的 23 份分数包含不同 K 和规则控制，不是 23 次独立训练；K4 与 K32 是嵌套样本预算。历史、synthetic、当前三种子及探索性两种子有独立的证据层，不能混表。真实 repeat、seed、实际地图数和数据字典都随包保存。

## 如何查论文数字

1. 打开[行目录](results/paper_submission_v1/evidence/row_catalog.json)，按 cohort、method、K、seed 找行，获取 snapshot JSON Pointer。
2. 用[数值索引](results/paper_submission_v1/evidence/numeric_index.csv)找未舍入值；所有目录项保留原字段，没有为凑表填入缺失方法。
3. 用[路径映射](results/paper_submission_v1/evidence/path_mapping.json)找到对应原报告在包内的路径及 SHA；`sources/` 文件逐字节保留。
4. 需要逐事件复核时，合并包内 `targets.csv` 与原始 `predictions_k*.csv`，主键为 `(global_id, candidate_index, radius_cells)`；也可直接运行下述脚本。

例如，从项目目录运行：

```bash
python results/paper_submission_v1/evidence/lookup.py --pointer /evidence_rows/1
python results/paper_submission_v1/evidence/lookup.py --search /metrics/brier/mean
python results/paper_submission_v1/evidence/verify_offline.py
```

脚本只依赖 Python 标准库；不需要 PyTorch、NumPy、GPU 或网络。将整个 `evidence/` 复制到其它目录后，仍可直接运行里面的 `verify_offline.py`。需要保存新的核验回执时加 `--report <尚不存在的文件路径>`；脚本拒绝覆盖既有回执。

## 实际核验结果

[离线核验回执](results/paper_submission_v1/evidence/verification.json)已通过：

- 153 个库存项的 SHA/大小、140 个原样文件和 44 个原报告 JSON Pointer 均匹配。
- 23 份保存分数使用同一组 1545 事件；物理 archive train 的 100/25/40 分区元数据与父地点无交叉条件匹配。
- 对每个报告重算 equal-parent、pooled 两种权重下的 Brier、NLL、ECE、false-safe@0.8、coverage@0.8、risk30，以及正例率与平均分数，共 **368 项**；与已冻结 snapshot 的最大绝对差为 **2.22×10⁻¹⁵**。
- 23×515=11,845 个查询半径三元组的原分数和标签均符合半径单调性；无新后处理。
- 24,272 条精确数字索引均回指相同原值；没有重新 bootstrap、选阈值、选种子或改查询。

| 对象 | SHA-256 |
|---|---|
| 原 snapshot JSON | `8a3b81ef1f01dec72e57f92d7c02d39ed0e3697e12389143a755a836cc4bbc32` |
| 库存 `inventory.json` | `7c7f51d790ef06587282d2c78ac8392b7330b8ed883301c9155a7c6733ffc487` |
| 本次 `verification.json` | `63163e3c61a0098529c3ce6bd38594123cc3e9ebce2638f22040d4d911e59b37` |
| 导出标量标签 CSV | `9f0115ff98be603576d3263cd1cb555f6d40b30553e892de7116fb4952b4e48d` |

库存不把自身、库存 SHA 文件或本次核验回执循环纳入哈希；这三个文件是总目录 156 与库存 153 之差。完整库存固定包内的正文/表格封存版；之后主工作区增加导航链接不会重写这个封存版。

## 读取边界与剩余缺口

[标签导出回执](results/paper_submission_v1/evidence/sources/results/paper_submission_v1/analysis/target_export_receipt.json)记录了本轮唯一涉及 NPZ 的操作：root 对 snapshot 已登记且先前已用过的 40 个 physical-train development-validation 缓存，仅展开已有 `candidate_indices`、`targets` 两数组导出标签；没有展开地图数组、读原始图像、生成新几何标签或访问其它 NPZ。本证据打包器和离线核验器本身没有读取任何 NPZ/权重，离线包也不含这些资产。

地图 Brier 必须区分二值世界频率与连续条件概率。包里只有相应旧聚合/逐案例指标与审计，**不能从本包重算栅格 Brier/IoU、网络输出或 footprint 几何**。独立核验的是已有标量事件预测的算术，不是重新执行模型实验。

新划分 direct-query/no-event/no-global 仍缺失；25 个 calibration 地点逐查询预测仍缺失；最终测试仍未开展。历史子场景重叠、历史物理 test 访问和当前验证复用都保留在原更正报告与[数据字典](results/paper_submission_v1/evidence/DATA_DICTIONARY.md)，没有通过复制文件修复成“独立最终测试”。外部模型未充分收敛，不进入正式主比较。

本包解决的是**已有论文证据的收集、查数和离线事件指标复核**。它不声称补齐上述实验，也不授权重启训练或打开最终测试集。
