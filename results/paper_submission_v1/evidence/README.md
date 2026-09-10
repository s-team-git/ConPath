# ConPath 现有论文数据离线包

本包整理已有 development-validation 证据，不新增训练、推理或最终测试。建议先读 `DATA_DICTIONARY.md`，再用 `row_catalog.json` 定位 29 行聚合数字；`path_mapping.json` 将原论文的路径映射到本包 `sources/` 内的原样文件。

直接核验（无需 PyTorch / NumPy / GPU / 网络）：

```bash
python verify_offline.py
```

从任意目录也可以运行脚本的绝对路径。生成新回执时指定尚不存在的输出路径；已有回执不会被覆盖：

```bash
python verify_offline.py --report /tmp/conpath_new_offline_verification.json
```

精确查数，不进行重新选模或改指标：

```bash
python lookup.py --pointer /evidence_rows/1
python lookup.py --search /metrics/brier/mean
```

`inventory.json` 与 `inventory.sha256` 固定整个包的非回执文件，`verification.json` 保存这次执行的实际核验结果。`numeric_index.csv` 展开未舍入的原始数值，图表舍入和原精度可分别审阅。

当前 scalar targets 与保存分数足以离线重算原 23 个方法/seed/预算报告的 Event Brier、NLL、ECE、false-safe@0.8、coverage@0.8、risk30（equal-parent 和 pooled）。地图数组与权重没有打包；不能靠此目录重跑网络、地图质量或几何评价。历史 cohort 只封存旧聚合/审计报告，不复制可能涉及历史物理 test 的逐事件标签。

当前主模型保留原版 ConPath 的 20260910、20260911、20260912。新划分缺失的 direct-query/no-event/no-global 没有被伪造为已完成。原始论文文件在 `sources/` 里是封存版本；本包是数字证据组件，论文 PDF 与图片请见上层投稿整理目录。
