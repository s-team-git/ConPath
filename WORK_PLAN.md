# ConPath 当前工作计划：论文收敛

2026-09-10用户最新指令优先于所有旧实验路线。

当前只做已有结果审计、英文论文、表格、真实图片、checkpoint封存和版本管理。外部模型训练与自动启动已取消；不扩数据、不加架构、不打开最终测试、不重复已完成实验。

三个问题：RQ1固定经验边际下的空间联合结构；RQ2 reachability训练监督；RQ3 footprint半径、遮挡、瓶颈与不可达事件可靠性。对应证据的协议、缺失项和限制必须分开写清。

停止条件已满足：父级隔离的原ConPath三个已完成seed均优于independent的Event Brier，因而不启动A/B/C，不拟合新温度，不做更多参数搜索。原版三seed是主方法；coherent两seed仅为补充探索。

交付：PAPER_DRAFT.md、PAPER_EVIDENCE.md、PAPER_TABLES.md、PAPER_FIGURES.md、results/paper_validation_snapshot.json/.csv、results/paper_figures/、results/FINAL_MODEL_SELECTION.md。

下一步在这些文件内部完成作者审阅、表达和投稿排版。最终测试先做不读测试资产的历史访问与留出资格审核，未经新授权不解锁。缺失的新协议消融不得从旧cohort移植，不自动训练补齐。

原计划保留在Git历史与 `results/paper_convergence_v1/prior_documents/`。外部队列的STOP必须保持，旧恢复/续训命令不再授权执行。
