# ConPath 英文论文排版工作稿

- [双栏主文 PDF](build/main.pdf)
- [补充材料 PDF](build/supplement.pdf)
- [编译记录](build/build_report.json)

正文基于项目根目录的 `PAPER_DRAFT.md`，围绕 RQ1/RQ2/RQ3 组织；`supplement.tex` 收录完整现有表、协议差别、缺失项、历史机制诊断、两种子探索和十个固定定性案例。它们是 **validation-only 工作稿**，不是正式测试报告，不声称满足尚未核验的某届 IROS/ICRA 页限或投稿条件。作者和单位尚未提供，仅标记匿名工作稿。

## 编译

已有本地 Tectonic 时，在仓库根目录运行：

```bash
python paper/build.py
```

若没有编译器，可运行 `python paper/bootstrap.py`。该脚本只下载 SHA-256 锁定的 Tectonic 0.17.0 Linux x86_64 便携二进制到 `paper/tools/`，不修改系统；压缩包约 10 MB。首次编译会按需下载 TeX 宏包/字体到 `paper/.cache/`。编译器及缓存不纳入 Git。其他平台可自行使用对应 Tectonic；亦可在 `paper/` 中用现有标准 LaTeX/BibTeX 工具链编译。系统 `pdfinfo` 用于生成页数记录。

`build.py` 只读取已有摘要、LaTeX 和已保存 PDF 图，生成排版文件；不导入 PyTorch、不打开数据集或 checkpoint、不启动训练或模型推理。`prepare_assets.py` 将根目录现有表以及已有数值快照的分层 CSV 转成表格，精确记录来源 SHA 和表格内容，不重新选择模型或重算预测。更改根文档后重新运行会更新 `table_provenance.json` 的来源版本。

## 来源和界限

- `table_provenance.json` 记录根草稿、证据表、模型选择和不可变 snapshot 的 SHA，以及全部表格的来源/显示值。当前结果、历史有重叠结果、两种子候选各自分开；缺失结果保留缺失。
- `figures/provenance.json` 记录 20 份已保存 PDF 图的原路径和 SHA，图像内容逐字节相同。主文中的英文网络流程框图是已有 `01_architecture` 代码示意的压缩翻译，不是测量图或新模型。两张主文数据图保留原中文标签并在英文 caption 完整解释；发布投稿版之前仍需不改变数据的英文图标注整理。
- `references.bib` 使用论文/作者/出版社一手来源核验的九条文献；没有填入不兼容外部论文的实验数字。FlatLands 明确为 arXiv v3/under review。
- 主文与补充均保留：validation-only、当前 direct-query/no-event/no-global 缺失、旧 27/160 观测父地点重叠、物理 test 历史访问、半径 20 时 ConPath 零高置信覆盖率、deterministic/radius-prior 反例、训练预算与梯度限制。已取消的外部训练仅作为未充分收敛的补充记录。
- 这里只完成文稿与排版。主张充分性和最终 holdout 资格不能从编译成功推导出来；测试仍锁定。

## 文件与许可

IEEEtran 1.8b 类文件及 BibTeX 样式来自 [CTAN IEEEtran](https://ctan.org/pkg/ieeetran)，保持原字节和内置许可头，采用 LPPL 1.3；见 `vendor/IEEEtran_README`、`vendor/IEEEtran.cls`、`vendor/IEEEtran.bst` 与 `vendor/provenance.json`。根目录 `IEEEtran.cls` 和 `IEEEtran.bst` 是指向 vendor 的相对软链接。Tectonic 来源见 [官方安装说明](https://tectonic-typesetting.github.io/en-US/install.html)，下载版本及压缩包 SHA 记录在 provenance 和 bootstrap 中。

`main.tex` / `supplement.tex` / `references.bib` 是可编辑源码；`tables/` 由摘要自动生成；`gallery.tex` 固定包含全部十例。`build/` 保留两份 PDF、编译日志及机器核验结果。`review/` 保留实际页面图像检查记录。中间 aux、缓存及便携二进制不提交。
