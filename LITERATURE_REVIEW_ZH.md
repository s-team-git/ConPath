# 相关论文如何做对比实验

核查日期：2026-09-07。阅读了下面八篇论文的实验章节、对应表格和必要的补充材料，
并检查了 CogniPlan、MapEx、UPEN、FlatLands 的官方仓库。本文记录原论文的做法；
我们据此制定的方案见 [EXPERIMENT_DESIGN_ZH.md](EXPERIMENT_DESIGN_ZH.md)。
本轮只阅读论文与代码，没有运行外部模型、下载新的实验数据或恢复训练。

## 结论

**主对比应围绕“部分二维地图 → 多种完整地图 → 通路概率”，优先比较地图补全方法。**
此前优先考虑 KITTI-360，是从三维场景补全文献出发；读到更贴近任务的地图论文后，
当前建议调整为：FlatLands 上做强补全基线，CogniPlan 原生地图上做外部模块比较，
KITTI-360 留作明确需要三维传感器实验时的扩展。

公开方法可以适配到新任务。有效适配需要保留其核心网络或算法，说明改了哪些接口，
在统一协议下实际运行。我们以前的 TinyBEV 坐标头和小网络集成，仍只能叫本地对照；
它们不能替代对 S4C 或 PaSCo 原网络的比较。

## 八篇论文的具体做法

| 论文与核查位置 | 原实验的数据、输入与对照 | 比较方式与我们应借鉴的内容 |
|---|---|---|
| [FlatLands，2026，v3](https://arxiv.org/html/2603.16016v3)，§5、表2–5，S5/S8.2 | 统一 BEV 观测和未知掩码；U-Net、PConv、LaMa、LaMa 集成、扩散、流匹配等；ID 与 ScanNet++ OOD 分开 | 主要定量实验比较补全阶段，RGB 前端另评。保留外部骨干并适配通道；统一未知有效区域。图像精度、生成分布和耗时分别报告。我们应优先采用这一层面的比较。 |
| [CogniPlan，CoRL 2025](https://proceedings.mlr.press/v305/wang25d.html)，§3.4、§4，表1–2、图4 | 模拟地图：150 张探索、100 张导航；探索比较 Nearest、NBVP、TARE Local、ARiADNE+ 等，导航比较 BIT、D*Lite 等；KTH 另比较 MapEx | 不同任务使用不同对照和指标，主要看行驶距离；KTH 的迁移设置单列。补全器可单独抽出，但其通路概率不能冒充原论文完整规划器的成绩。 |
| [MapEx，ICRA 2025，v3](https://arxiv.org/abs/2409.15590v3)，§IV–V、图7–8 | KTH 楼层图；按建筑隔离训练/测试；10 张保留图、每图4个起点。对比 Nearest、UPEN、IG-Hector | 在共同模拟器中比较覆盖率、障碍物 IoU、规划成功率。主比较与消融分开；仓库明确说明 UPEN/IG-Hector 是作者重实现。说明“有记录的任务适配”可以是正式对照。 |
| [UPEN，ICRA 2022](https://arxiv.org/abs/2202.11907)，§IV、表I–III | Matterport3D/Habitat，深度输入；比较 ANS、OccAnt、FBE，以及与 DD-PPO 组合的导航方案 | 使用标准场景划分，区分噪声设置；部分实验共用地图预测与局部策略，隔离上层规划作用。它的固定候选路径评分与我们的“任意通路存在概率”不是同一个预测量。 |
| [PaSCo，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cao_PaSCo_Urban_3D_Panoptic_Scene_Completion_with_Uncertainty_Awareness_CVPR_2024_paper.html)，§4、表1–3 | 三维 LiDAR：SemanticKITTI、SSCBench-KITTI360，另做 Robo3D 扰动；新 PSC 任务组合 SSC 网络与 MaskPLS，并重新训练 | 主方法表与不确定性表分开。后者在同一骨干上实现 TTA、MC Dropout、Deep Ensemble，并报告参数、前向次数和时间。我们应区分“完整方法比较”和“相同骨干的机制比较”。 |
| [S4C，3DV 2024](https://arxiv.org/abs/2310.07522v2)，§4、表1、附录C.2–C.3 | KITTI-360 视频自监督训练；SSCBench-KITTI360 上与 MonoScene、LMSCNet、SSCNet 比较 | 明确监督与模态差异；修正有效区域后，用公开权重重新评估所有方法。未能可靠复现的 VoxFormer 没有被当作弱基线展示。我们也应先解决适配故障再比较。 |
| [SGN，TIP 2024，v2](https://arxiv.org/abs/2312.05752v2)，§IV | 相机三维补全：SemanticKITTI、SSCBench-KITTI360，另有 NYUv2；比较 MonoScene、VoxFormer、OccFormer 等 | 使用任务共同的 IoU/mIoU，标注输入类型，报告不同距离和模型规模；验证集选检查点。适合借鉴效率和分层结果的报告方式，当前不优先移植整个三维系统。 |
| [SceneSense，IROS 2024](https://arxiv.org/abs/2403.11985v1)，§V、表I | HM3D：10 栋训练、2 栋测试；RGB-D 和运行中的三维地图；主要对照为观测 OctoMap | 新任务下采用较窄的观测基线，报告 FID/KID 等。可借鉴保持观测不变的约束，不能据此推导我们只需内部消融；当前已有更贴近二维任务的公开对照。 |

## 原文和官方代码核查带来的限制

### FlatLands：最接近当前任务，但不能直接搬运表中数字

其主要定量输入也是 BEV，而不是要求每个补全器直接读 RGB。我们此前对前端差异的强调不够准确。
目前官方仓库仍说明模型权重和工具待发布；原文主表的随机模型损失描述与 S5 的 MSE 方程存在不一致。
因此，条件生成对照需逐项记录数学定义和实现选择，不能称为已复现作者分数。
另有尺度差异：v3 写约 0.039 m/格，本地既有审计记录元数据为 0.01 m/格。
这尚未证明哪一方错误；正式物理半径实验应先核对生成、裁剪和缩放记录。
[论文](https://arxiv.org/html/2603.16016v3)、[官方发布状态](https://github.com/1ssb/Flat_Lands)。

### CogniPlan：公开模块很合适，但不能直接替换训练文件

官方补全数据加载器按文件名读取 room/tunnel/outdoor 标签，作为三维条件向量；
FlatLands 没有这三类标签。故优先在其原生地图数据上比较，避免任意编造布局类别。
官方评估器使用4个布局条件；原论文规划器训练用7个、主结果推理用4个。
这些是有限的布局假设，不是4次独立后验采样，通路投票需要作为预测分数检验校准。
原生后处理还含阈值与形态学运算，必须记录其对窄通道的影响。
[官方代码](https://github.com/marmotlab/CogniPlan)。

### MapEx：LaMa 预测器和完整探索方法要分别命名

官方仓库有预测器加载代码、LaMa 子模块和 KTH 权重链接。完整方法用一个主预测器与3成员集成；
FlatLands 文献则采用4成员 LaMa 集成。我们主表拟采用后者的共同采样设置，准确标注为
“LaMa 集成：BEV 适配”，而非“完整 MapEx”。不同论文的成员数、微调数据和预训练来源不能混写。
[官方代码](https://github.com/castacks/MapEx)。

## 我们采用的比较原则

1. **先确定预测任务，再选数据与论文。** 外部方法、强简单基线和消融承担不同作用。
2. **实际重新计算同一指标。** 论文原有 mIoU、距离或最佳样本成绩不进入通路概率主表。
3. **保留外部方法的实力。** 不把大骨干替换为 TinyBEV 后继续沿用原论文名称；不将适配失败作为优越性证据。
4. **条件、范围与成本要可见。** 输入信息、有效区域、额外训练标签、预训练、采样数、参数量和时间都记录。
5. **适配方法与原生系统分开。** 地图模块比较支持地图/事件结论；完整导航优势需要独立的闭环实验。
6. **看不到答案时也能执行评估。** 不使用真实地图选最好样本，不按测试结果调阈值或删失败场景。

## 证据记录

完整 PDF、文本抽取、官方代码快照和 SHA-256 记录位于 results/literature_protocol_20260907/。
阅读的 PDF 版本为 FlatLands v3、MapEx v3、S4C v2、SGN v2、SceneSense v1、UPEN v1；
CogniPlan 使用 PMLR 正式版本，PaSCo 使用 CVPR 正式版本。
官方仓库快照：CogniPlan 444fab8d5d3b；MapEx 53636bd1c791；
FlatLands 6c4aabe54459；UPEN 17fc0390c45e。
本轮仅确认代码、发布入口和数据资产元信息；没有验证下载权重后的数值复现。
