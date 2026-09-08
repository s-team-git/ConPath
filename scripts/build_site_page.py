#!/usr/bin/env python3
"""Render a Chinese project page from the frozen numerical/visual snapshots."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'


def zoom(path, caption, eager=False):
    return f'<a class="zoomable" href="{html.escape(path)}" data-zoom data-caption="{html.escape(caption)}"><img src="{html.escape(path)}" alt="{html.escape(caption)}" loading="{"eager" if eager else "lazy"}" decoding="async"><span class="zoom-hint" aria-hidden="true">点击放大 ↗</span></a>'


def main():
    data=json.loads((SITE/'data/site_visuals_zh.json').read_text())
    analysis=json.loads((SITE/'data/flatlands_clean_paper_analysis.json').read_text())
    case=data['examples'][0]
    panels=''.join(zoom(case['panels'][key],caption,True) for key,caption in [('observed','① 已观测地图：绿色可通行，深灰阻挡，浅灰未知。S 为起点，G 为目标。'),('correlated','② ConPath 推测：米白至青绿表示单元格可通行概率从 0 到 1。'),('correlated_sample','③ 第一次随机补全的完整世界：绿色可通行，深灰阻挡。'),('reference','④ 数据集完整参考地图，用于核对通路是否存在。')])
    first=data['gallery']['flatlands'][0]
    gallery=zoom(first['observed'],'训练集中的已观测地图')+zoom(first['reference'],'同一场景的完整参考地图')
    gallery_source=html.escape(f"训练集 · {first['source']} / {first['scene']} · {first['id']} · 每格 {first['resolution_m']:.2f} 米。模型输入与完整参考严格区分。")
    thumbs=''.join(f'<button class="thumbnail {"selected" if i==0 else ""}" type="button" data-gallery-index="{i}" aria-label="查看 {html.escape(row["source"])} 场景 {i+1}" aria-pressed="{"true" if i==0 else "false"}"><img src="{row["observed"]}" alt="" role="presentation" loading="lazy"><span>{html.escape(row["source"])}</span></button>' for i,row in enumerate(data['gallery']['flatlands']))
    frame=data['gallery']['sequence'][0]
    sequence=f'<figure><div class="camera-stage">{zoom(frame["camera"],"同一时刻的原始相机照片")}</div><figcaption>① 相机原始画面<span>帮助理解场景；当前模型输入来自激光雷达</span></figcaption></figure>'+''.join(f'<figure>{zoom(frame[key],label)}<figcaption>{label}</figcaption></figure>' for key,label in [('observed','② 激光雷达观测'),('reference','③ 数据集参考地图')])
    rows=[]
    for key,method in analysis['methods'].items():
        cells=[html.escape(data['method_labels'][key]),str(method['seed_count'])]
        for metric in ('brier','nll','ece'):
            v=method['aggregate'][metric];cells.append(f'{v["mean"]:.5f}'+(f' ± {v["sample_sd"]:.5f}' if v['sample_sd'] is not None else ''))
        cells.append(f'{method["equal_coverage"]["0.3"]["mean"]*100:.2f}%')
        row='<tr'+(' class="ours"' if key=='conpath' else '')+f' data-method="{key}">'
        row+='<th scope="row">'+cells[0]+'</th>'+''.join('<td>'+v+'</td>' for v in cells[1:])+'</tr>'
        rows.append(row)
    page='''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#ffffff">
  <meta name="description" content="ConPath 中文研究主页：用真实地图、中文图例和数据示例，理解未知空间中的路径概率、实验结果与局限。">
  <title>ConPath · 看懂未知空间中的路径概率</title>
  <link rel="icon" href="favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="styles-zh.css"><script src="app-zh.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#method">跳到正文</a>
  <header class="site-header"><nav class="shell nav" aria-label="主导航"><a href="#top" class="wordmark">ConPath<span>研究笔记</span></a><div class="nav-links"><a href="#method">看懂方法</a><a href="#datasets">数据示例</a><a href="#results">实验结果</a><a href="#comparison">数据集与对照</a></div></nav></header>
  <main id="top">
    <section class="hero shell">
      <p class="eyebrow">部分观测 · 地图不确定性 · 路径概率</p>
      <h1>ConPath<span>看不见的空间，<br class="mobile-break">还有路可走吗？</span></h1>
      <p class="hero-description">机器人只看到了地图的一部分。我们让模型补全多种可能的世界，<br class="desktop-break">再考虑机器人的尺寸，估计起点到目标之间<strong>存在通路的概率</strong>。</p>
      <div class="publication-links"><a class="pill primary" href="https://github.com/s-team-git/ConPath/blob/main/PAPER_DRAFT.md">论文草稿（英文）↗</a><a class="pill" href="https://github.com/s-team-git/ConPath">项目代码 ↗</a><a class="pill" href="#method">先看图，理解方法 ↓</a></div>
      <p class="status-line"><span class="status-dot"></span>验证阶段 · 新消融训练已恢复 · 更新于 2026.09.07</p>
    </section>

    <section id="method" class="section shell">
      <div class="section-heading"><p class="section-number">01 / 方法</p><h2>从一张不完整的地图，<br>到一个关于通路的概率。</h2><p>下面是已训练模型的真实验证示例。按 ① → ④ 阅读；每张图都可以放大。</p></div>
      <div class="toolbar"><div class="segmented" role="group" aria-label="选择读图示例"><button type="button" class="selected" data-example="0" aria-pressed="true">示例一：参考地图有路</button><button type="button" data-example="1" aria-pressed="false">示例二：参考地图无路</button></div><label class="model-select">查看模型 <select id="model-select"><option value="correlated">ConPath：相关补全</option><option value="independent">独立单元对照</option></select></label></div>
      <div class="mobile-steps" role="group" aria-label="选择读图步骤"><button type="button" data-step="0" class="selected" aria-pressed="true">① 观测</button><button type="button" data-step="1" aria-pressed="false">② 推测</button><button type="button" data-step="2" aria-pressed="false">③ 补全</button><button type="button" data-step="3" aria-pressed="false">④ 参考</button></div><div id="example-panels" class="example-panels">{{PANELS}}</div>
      <div class="map-legend" aria-label="地图颜色与标记说明"><span><i class="swatch free"></i>可通行</span><span><i class="swatch blocked"></i>阻挡</span><span><i class="swatch unknown"></i>尚未观测</span><span><i class="swatch outside"></i>有效范围外</span><span><b class="start-symbol">● S</b>起点</span><span><b class="goal-symbol">◆ G</b>目标</span></div>
      <p class="reading-note">概率图单独使用图内的 0–1 色条，颜色越深，单元格越可能可通行。<strong>S 和 G 只是查询的两个端点，图中没有把它们画成一条规划路线。</strong></p>
      <div class="example-summary" aria-live="polite"><div><span id="example-verdict">参考地图：存在通路</span><strong id="example-probability">ConPath 预测通路概率 78.9%</strong></div><p id="example-explanation">把可能的完整地图逐一检查后，模型给出“有路”的概率。中间的均值概率图描述每个位置，最终通路概率还取决于这些位置能否共同连通。</p></div>
      <p id="example-source" class="provenance">FlatLands / ZInD · obs_266762 · 机器人半径 10 格 · 模型种子 20260831。按验证标签挑选的解释性示例，不能用两个例子代替整体统计。</p>
      <details class="plain-details"><summary>“一次可能的世界”和“概率地图”有什么区别？</summary><div class="detail-body"><p>概率地图为每一个格子给出可通行概率，例如 70%。一次随机补全则把整张地图变成一个具体世界：每格要么可通行，要么阻挡。两张概率地图即使看起来接近，也可能产生连通性很不一样的完整世界。</p><p>为考虑机器人的大小，我们先从可通行区域边缘扣除一个圆盘半径，再用四邻接关系检查起点和目标是否连通。这里评估的是<strong>通路是否存在</strong>，尚未输出带车辆动力学约束的行驶轨迹。</p></div></details>
    </section>

    <section id="datasets" class="section section-wash"><div class="shell">
      <div class="section-heading"><p class="section-number">02 / 数据</p><h2>先看清楚，模型到底看到了什么。</h2><p>6 个室内地图示例、6 个户外场景，以及一段 18 帧的真实数据浏览。<br>新增图库均取自训练集，场景按来源和编号选取，没有按模型表现挑图。</p></div>
      <div class="dataset-tabs segmented" role="group" aria-label="切换数据集"><button type="button" data-dataset="flatlands" class="selected" aria-pressed="true">FlatLands · 室内地图</button><button type="button" data-dataset="unscenes3d" aria-pressed="false">UnScenes3D · 户外实景</button></div>
      <div class="gallery-heading"><div><h3 id="gallery-title">FlatLands · 3RScan</h3><p id="gallery-description">左边是模型能够看到的部分，右边是用于核对的完整参考地图。这些是俯视栅格图，不是相机照片。</p></div><span id="gallery-count" class="counter">01 / 06</span></div>
      <div id="gallery-stage" class="gallery-stage">{{GALLERY}}</div>
      <div class="map-legend"><span><i class="swatch free"></i>可通行</span><span><i class="swatch blocked"></i>阻挡</span><span><i class="swatch unknown"></i>未知</span><span><i class="swatch outside"></i>有效范围外</span></div>
      <div id="gallery-thumbnails" class="thumbnails" aria-label="选择具体场景">{{THUMBNAILS}}</div>
      <p id="gallery-source" class="provenance">{{GALLERY_SOURCE}}</p>
      <div class="sequence-intro"><h3>同一时刻：相机照片 → 观测地图 → 参考地图</h3><p>这是一段 UnScenes3D 原始数据的逐帧浏览。照片是前视图，地图是俯视图；当前模型读取激光雷达生成的观测地图，参考地图用于监督和评估。</p></div>
      <div id="sequence-stage" class="sequence-stage">{{SEQUENCE}}</div>
      <div class="sequence-controls"><button id="sequence-play" type="button" class="pill small" aria-pressed="false">播放浏览</button><label for="sequence-frame" class="sr-only">选择数据帧</label><input id="sequence-frame" type="range" min="0" max="17" value="0" step="1"><output id="sequence-counter" for="sequence-frame">01 / 18</output></div>
      <p class="reading-note">每秒切换一张采样帧，仅用于看数据，并非原始录像速度或模型实时运行速度。<strong>没有绘制查询直线或预测行驶路线。</strong></p>
      <p class="source-line">来源：<a href="https://1ssb.github.io/Flat_Lands/">FlatLands 官方项目</a> · <a href="https://github.com/ruiqi-song/UnScenes3D">UnScenes3D 官方数据</a> · <a href="data/site_visuals_zh.json">每张图的场景、时间戳与来源记录</a></p>
    </div></section>

    <section id="results" class="section shell">
      <div class="section-heading"><p class="section-number">03 / 结果</p><h2>哪些结果有支持，哪些还没有？</h2><p>当前主要证据来自 FlatLands：142 个验证场景，每次训练评估相同的 4,224 个路径事件。<br>场景之间等权，正式测试集尚未评估。</p></div>
      <div class="findings"><div><span>路径概率误差 ↓</span><strong>0.0675 <small>vs 0.0952</small></strong><p>ConPath 比独立单元对照低约 29.1%。</p></div><div><span>强确定性对照</span><strong>0.0696</strong><p>同模型的均值地图已很接近 ConPath。</p></div><div><span>相同 30% 覆盖率的误判风险</span><strong>3.60% <small>vs 3.90%</small></strong><p>配对区间包含零，尚无稳定风险优势。</p></div></div>
      <div class="chart-toolbar"><div class="chart-tabs" role="group" aria-label="选择结果图"><button type="button" data-chart="brier" class="selected" aria-pressed="true">概率误差</button><button type="button" data-chart="risk" aria-pressed="false">误判风险</button><button type="button" data-chart="reliability" aria-pressed="false">概率可信度</button><button type="button" data-chart="sampling" aria-pressed="false">采样次数</button><button type="button" data-chart="dependence" aria-pressed="false">空间相关性</button></div><a id="chart-pdf" href="assets/zh/brier.pdf" class="download-link">下载 PDF ↗</a></div>
      <figure class="main-chart"><a href="assets/zh/brier.svg" id="chart-open" data-zoom data-caption="路径概率的预测误差：越低越好"><picture><source id="chart-mobile" media="(max-width: 600px)" srcset="assets/zh/brier-mobile.svg"><img id="chart-image" src="assets/zh/brier.svg" alt="中文横向柱状图：比较八种本地对照的路径事件 Brier 分数，越低越好，误差线为训练种子标准差。" loading="lazy"></picture></a><figcaption id="chart-caption"><strong>怎么看：</strong>Brier 衡量预测概率与实际结果的偏差，0 最好。横线表示三次训练的标准差；它不是置信区间。所有对照使用相同的验证查询。</figcaption></figure>
      <details id="paper-analysis" class="plain-details"><summary>展开完整数值表与指标解释</summary><div class="detail-body"><div class="table-scroll"><table class="metrics-table"><caption>相同验证查询的九种本地对照 · 均值 ± 训练种子标准差</caption><thead><tr><th scope="col">方法 / 对照</th><th scope="col">种子数</th><th scope="col">Brier ↓</th><th scope="col">NLL ↓</th><th scope="col">ECE ↓</th><th scope="col">30% 覆盖率误判 ↓</th></tr></thead><tbody>
<!-- CLEAN_PAPER_ROWS_START -->
{{ROWS}}
<!-- CLEAN_PAPER_ROWS_END -->
      </tbody></table></div><p><strong>Brier：</strong>概率的平方误差。<strong>NLL：</strong>对错误且过度自信的判断惩罚更重。<strong>ECE：</strong>模型信心与实际频率的分箱差异。<strong>误判通路：</strong>接受的查询里，实际上没有通路的比例。种子数代表独立训练次数；半径先验只由训练集拟合一次。</p><p>均值地图等二元预测会出现同分查询，覆盖率边界按不使用标签的比例方式分配。<a href="data/flatlands_clean_paper_analysis.json">原始统计 JSON ↗</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/PAPER_EVIDENCE.md">完整证据与复现说明 ↗</a></p></div></details>
      <div class="second-domain"><h3>换到户外数据后，当前还没有看到优势。</h3><p>UnScenes3D 的均值地图事件误差为 <strong>0.51142</strong>（ConPath）与 <strong>0.51130</strong>（独立对照），结果几乎相同。当前硬观测规则已经造成约 <strong>0.47574</strong> 的误差下界；只加训练时长难以解决，需要先检查观测模型。这里只有两个验证场景，尚不支持跨域成功的结论。</p><p class="provenance">这里比较的是均值地图事件，预测对象与上方随机路径概率不同。<a href="data/unscenes3d_clean_support_k128_candidate.json">修正后的验证报告</a> · <a href="data/unscenes3d_observation_ceiling.json">观测误差下界</a></p></div>
    </section>

    <section id="comparison" class="section section-wash"><div class="shell text-shell">
      <div class="section-heading"><p class="section-number">04 / 研究选择</p><h2>为什么是这两个数据集？</h2></div>
      <div class="dataset-rationale"><article><span class="small-label">主实验 / 室内</span><h3>FlatLands</h3><p>它直接提供<strong>部分观测、完整俯视地面地图和有效范围</strong>，适合构造“这两个点之间，对指定尺寸的机器人是否有路”的标签。选它首先是因为与研究问题匹配。</p><p class="provenance">当前使用场景隔离的自建来源划分，不是官方排行榜划分。本地输入是地图通道，没有声称复现原论文从单张 RGB 图像出发的完整系统。<a href="https://1ssb.github.io/Flat_Lands/">官方介绍 ↗</a></p></article><article><span class="small-label">第二域诊断 / 户外</span><h3>UnScenes3D</h3><p>它提供<strong>真实车辆相机、激光雷达和三维占用标注</strong>，可检查从室内地图换到户外观测时，方法哪里会失效。标注可投影为保守的可通行地面图。</p><p class="provenance">目前使用公开 mini 数据中的 9 个训练场景、2 个验证场景，测试地点 location_6 未使用。它与常见的 nuScenes 是两个不同数据集。<a href="https://www.nature.com/articles/s41597-025-05532-5">数据论文 ↗</a></p></article></div>
      <h3 class="comparison-title">读过相关论文后，我们准备怎样比较？</h3><p>已核查八篇论文的实验章节与相关官方代码。主实验优先比较<strong>地图补全方法</strong>，再用共同的通路判断评估；外部方法比较与消融都需要。下面的新增对照仍是计划，尚未运行。</p>
      <div class="table-scroll"><table class="literature-table"><caption>原论文的实验设置与当前对照选择</caption><thead><tr><th scope="col">原论文</th><th scope="col">原论文的数据与比较方式</th><th scope="col">本项目的采用方式</th></tr></thead><tbody>
        <tr><th scope="row"><a href="https://arxiv.org/html/2603.16016v3">FlatLands · 2026 ↗</a></th><td>统一 BEV 输入；U-Net、LaMa/集成、扩散与流匹配</td><td>主实验优先增加 LaMa/集成和条件生成强对照；官方生成模型工具仍待发布。</td></tr>
        <tr><th scope="row"><a href="https://github.com/castacks/MapEx">MapEx · ICRA 2025 ↗</a></th><td>KTH 楼层图；比较 Nearest、UPEN、IG-Hector</td><td>使用公开 LaMa 补全代码；补全模块的比较不等于完整探索系统比较。</td></tr>
        <tr><th scope="row"><a href="https://proceedings.mlr.press/v305/wang25d.html">CogniPlan · CoRL 2025 ↗</a></th><td>模拟地图及 KTH；分别评估探索与导航</td><td>优先在其原生地图上比较官方生成模块，保留原有布局条件。</td></tr>
        <tr><th scope="row"><a href="https://github.com/astra-vision/PaSCo">PaSCo · CVPR 2024 ↗</a></th><td>SemanticKITTI、SSCBench-KITTI360</td><td>借鉴集成不确定性思路；旧集成对照尚未形成当前修正协议下的最终基线。</td></tr>
        <tr><th scope="row"><a href="https://github.com/ahayler/s4c">S4C · 3DV 2024 ↗</a></th><td>KITTI-360；在 SSCBench-KITTI360 上评估</td><td>本地坐标查询对照借鉴其隐式查询思路，并非原版三维系统。</td></tr>
        <tr><th scope="row"><a href="https://github.com/Jieqianyu/SGN">SGN · TIP 2024 ↗</a></th><td>SemanticKITTI、SSCBench-KITTI360；另有 NYUv2</td><td>相关工作参考；尚未在本项目相同输入和任务上运行。</td></tr>
        <tr><th scope="row"><a href="https://arxiv.org/html/2409.10681v1">在线 SceneSense · 2024 ↗</a></th><td>作者采集的真实建筑占用地图</td><td>三维占用生成与探索任务；不能直接把其分数放进路径概率表。</td></tr>
      </tbody></table></div>
      <p class="research-decision"><strong>当前方案：</strong>FlatLands 做主比较，CogniPlan 原生地图补充外部生成模块比较，KTH 作为后续泛化补充。KITTI-360 暂不列为必跑。先核对数据尺度、原生实现和计算预算，再安排训练；所有方法统一起终点、机器人尺寸与评价规则，实验数值由实际运行产生。</p>
      <details class="plain-details"><summary>主实验具体比较什么？</summary><div class="detail-body"><p>第一张新表计划统一为 4 个完整地图输出，比较 ConPath、LaMa 集成、条件流匹配等；确定性方法保留单个输出。另一张表记录实际采样量、误差、耗时和显存。现有 128 次采样的数值不会直接混进这张新表。</p><p>实验方案已确认：补齐外部方法对比、三次独立训练、场景配对统计和失败案例，再写入论文。已知限制也会保留：均值地图结果很接近，外部强基线尚未运行，FlatLands 原文与本地元数据的尺度差异待核对。六组消融已恢复，当前没有新增的最终结果。</p></div></details>
      <p class="source-line"><a href="https://github.com/s-team-git/ConPath/blob/main/LITERATURE_REVIEW_ZH.md">八篇论文怎样做比较 ↗</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/EXPERIMENT_DESIGN_ZH.md">具体对比实验方案 ↗</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/DATASET_CHOICE_ZH.md">数据集选型分析 ↗</a></p>
    </div></section>
    <section class="section shell closing"><h2>目前做到哪一步？</h2><p>主验证结果与图表已完成。9 月 7 日晚已按要求恢复六组训练消融：三组从已保存的第 4 轮续训，另外三组自动排队。全部完成并核验后更新表格与图片。<br>论文仍是工作草稿；已确认的后续工作包括外部方法主对比、CogniPlan 原生地图实验、计算成本和统计分析。当前页面展示已完成的验证结果。</p><div class="resource-links"><a href="https://github.com/s-team-git/ConPath/blob/main/WORK_PLAN.md">中文工作计划 ↗</a><a href="https://github.com/s-team-git/ConPath/blob/main/EXPERIMENT_DESIGN_ZH.md">已确认的实验方案 ↗</a><a href="https://github.com/s-team-git/ConPath/blob/main/CLEAN_ABLATION_PLAN.md">消融方案 ↗</a><a href="archive/index.html">旧版页面与图表归档 ↗</a></div></section>
  </main>
  <footer class="shell footer"><p>ConPath · 可复现研究记录</p><p>页面结构参考 <a href="https://s-team-git.github.io/Lightweight-3DGS/">Lightweight-3DGS</a>，重新实现中文排版与交互。<br>FlatLands 派生地图保留各上游来源条款；UnScenes3D 数据按其 CC BY 4.0 使用说明署名。<a href="https://github.com/s-team-git/ConPath/blob/main/DATASET_CHOICE_ZH.md">数据来源与使用说明</a>。</p></footer>
  <dialog id="image-dialog" aria-labelledby="dialog-caption"><div class="dialog-toolbar"><p id="dialog-caption"></p><button type="button" id="dialog-close" aria-label="关闭放大图片">关闭 ×</button></div><div class="dialog-image"><img id="dialog-image" alt="放大后的当前图片"></div><a id="dialog-source" href="#top">打开原始尺寸 ↗</a></dialog>
  <p id="interaction-error" class="interaction-error" hidden>交互数据暂时加载失败；已展示首个示例与完整静态结果，请刷新重试。</p>
</body></html>
'''
    for key,value in [('PANELS',panels),('GALLERY',gallery),('GALLERY_SOURCE',gallery_source),('THUMBNAILS',thumbs),('SEQUENCE',sequence),('ROWS','\n'.join(rows))]:page=page.replace('{{'+key+'}}',value)
    assert '{{' not in page
    (SITE/'index.html').write_text(page)
    print('Chinese page built: 9 methods, 2 model examples, 12 gallery scenes, 18 frames.')


if __name__=='__main__':main()
