# ConPath 论文工作稿与实验数据包

这是基于已有实验的作者审阅版本，全部科学结果仍为 validation-only。它不是最终测试报告，也没有完成新的六方法正式消融矩阵。

## 从哪里开始

1. 打开 `paper/build/main.pdf` 阅读双栏英文主文。
2. 打开 `paper/build/supplement.pdf` 查看完整表格、协议、固定案例及限制。
3. 打开根目录 `PAPER_READING_ZH.md` 阅读中文解释；`PAPER_CLAIM_AUDIT.md` 逐项解释主张和缺口。
4. 实验数据入口为 `PAPER_DATA_PACKAGE.md` 和 `results/paper_submission_v1/evidence/README.md`。

数据包提供23份现有事件分数、1545个现有开发验证标签、29行冻结汇总、训练配置/选模/历史日志、276行种子与分层数据，以及来源文件的SHA-256。23份文件预测的是重复的同一套事件，不能把35535条分数记录当成35535个独立测试事件。

## 离线验证

在解压目录中执行：

```bash
python results/paper_submission_v1/evidence/verify_offline.py
```

核验脚本仅依赖Python标准库，读取证据包内部文件。它检查数据库存、预测/标签查询对齐，重算地点等权及pooled事件指标，并与封存结果逐项比较。无需GPU、模型权重、原始图像或网络。

地图质量指标在包中有原始报告和哈希，但包内不含原始地图/世界数组，不能据此独立重算地图几何。旧cohort只包含聚合机制诊断，不能通过拼接数字变成新划分的正式消融。

## 文件与复现范围

- `paper/`：LaTeX源码、BibTeX、IEEEtran类/样式、引用图及实际编译PDF；具体编译方式见其README。
- `results/paper_submission_v1/evidence/`：独立可携带的标量证据包，含库存、源路径映射、数字定位与离线核验。
- `results/paper_submission_v1/analysis/`：从已冻结的物理train开发缓存导出的事件标签、只从snapshot展开的表格视图及来源回执。导出脚本依赖原项目缓存，离线包应使用上面的核验脚本，无须重新导出。
- `results/paper_submission_v1/review/`：论文主张核对和仅基于既有元数据的持出资格审计。
- `results/paper_figures/`：已有真实结果的图和图源清单。

部分原始研究文档保留项目内来源路径，具体归档位置请查 `evidence/path_mapping.json`；未携带的权重和数组在 `excluded_sources.json` 中明确列出。原始脚本中的仓库绝对路径是历史来源，不是离线核验依赖。

目录不包含TeX编译器二进制、下载缓存、模型权重、原始地图或最终测试资产。使用项目本地编译脚本时，首次准备TeX依赖可能需要网络；直接阅读PDF和核验事件数据可完全离线。

为便于跨平台解压，ZIP中的两个IEEEtran类/样式软链接保存为与本地vendor文件逐字节相同的普通副本，许可头和随附说明保留。

## 不得改变的结论边界

选择原版ConPath的现有三个checkpoint，种子20260910、20260911、20260912。新的40地点开发集没有direct-query、no-event、no-global完整对照；旧27/160观测与训练父地点重叠的结果保持历史诊断标记。候选改进只有两个种子，不能混入三种子均值。

半径20格时ConPath置信度0.8覆盖率为零，误判率未定义；确定性模型在该半径Brier更好。历史访问审计尚未证明最终持出资格。外部训练保持停止，最终测试仍锁定。本包的整理和排版不改变这些事实。
