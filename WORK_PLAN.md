# ConPath 当前工作计划：论文收敛

2026-09-10用户最新指令优先于所有旧实验路线。

当前只做已有结果审计、英文论文、表格、真实图片、checkpoint封存和版本管理。外部模型训练与自动启动已取消；不扩数据、不加架构、不打开最终测试、不重复已完成实验。

三个问题：RQ1固定经验边际下的空间联合结构；RQ2 reachability训练监督；RQ3 footprint半径、遮挡、瓶颈与不可达事件可靠性。对应证据的协议、缺失项和限制必须分开写清。

停止条件已满足：父级隔离的原ConPath三个已完成seed均优于independent的Event Brier，因而不启动A/B/C，不拟合新温度，不做更多参数搜索。原版三seed是主方法；coherent两seed仅为补充探索。

交付：PAPER_DRAFT.md、PAPER_EVIDENCE.md、PAPER_TABLES.md、PAPER_FIGURES.md、results/paper_validation_snapshot.json/.csv、results/paper_figures/、results/FINAL_MODEL_SELECTION.md。

本轮继续完成投稿排版与实验数据封存：`paper/` 双栏英文主文/补充材料、`PAPER_READING_ZH.md` 中文导读、`PAPER_DATA_PACKAGE.md` 离线数据入口，以及 `PAPER_CLAIM_AUDIT.md` 的逐RQ证据核对。1545个既有标签与23份保存预测已组成可移植标量数据包；276行分层视图让半径/来源/正负事件的局限可查。最终交付状态见 `results/paper_submission_v1/completion.json`，不是新增训练任务。

下一步由作者审阅论文主张、失败案例与证据缺口。仅元数据的历史访问审计已确认持出资格仍未证明；1532父地点部分接触名单不能认证其余地点untouched。不创建新测试集合、不解锁。缺失的新协议消融不得从旧cohort移植，不自动训练补齐。

原计划保留在Git历史与 `results/paper_convergence_v1/prior_documents/`。外部队列的STOP必须保持，旧恢复/续训命令不再授权执行。
