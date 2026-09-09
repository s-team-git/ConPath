#!/usr/bin/env python3
"""Render a Chinese project page from the frozen numerical/visual snapshots."""
import html
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'


def zoom(path, caption, eager=False):
    return f'<a class="zoomable" href="{html.escape(path)}" data-zoom data-caption="{html.escape(caption)}"><img src="{html.escape(path)}" alt="{html.escape(caption)}" loading="{"eager" if eager else "lazy"}" decoding="async"><span class="zoom-hint" aria-hidden="true">点击放大 ↗</span></a>'


def training_ablations():
    path = SITE / 'data/flatlands_clean_training_ablations.json'
    if not path.is_file():
        return ''
    report = json.loads(path.read_text())
    visuals = json.loads((SITE / 'data/training_ablation_visuals_zh.json').read_text())
    if (not report['validation_only'] or report['test_evaluated'] is not False
            or visuals['report_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
            or len(report['audits']) != 6 or not all(r['passed'] for r in report['audits'])):
        raise ValueError('training ablations must have six successful audits and matching public sources')
    rows = []
    for key, method in report['methods'].items():
        cells = []
        for metric in ('brier', 'nll', 'ece'):
            value = method['aggregate'][metric]
            cells.append(f'{value["mean"]:.5f} ± {value["sample_sd"]:.5f}')
        risk = method['risk_at_30_percent']
        cells.append(f'{risk["mean"] * 100:.2f} ± {risk["sample_sd"] * 100:.2f}%')
        rows.append(f'<tr data-ablation="{key}"' + (' class="ours"' if key == 'conpath' else '') +
                    f'><th scope="row">{html.escape(visuals["labels"][key])}</th>' +
                    ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    conclusions = ''.join(f'<article><h4>{html.escape(visuals["labels"][key])}</h4><p>{html.escape(visuals["summary"][key])}</p></article>' for key in ('no_event', 'no_global'))
    risk_intervals = [row['equal_coverage_risk']['0.3']['bootstrap_95'] for comparison in report['paired'].values() for row in comparison['seeds']]
    risk_note = '但相同 30% 覆盖率下，六个误判风险区间都包含零，尚不能宣称安全性稳定提升。' if all(lo <= 0 <= hi for lo, hi in risk_intervals) else '路径概率误差与误判风险分别评价；完整的逐种子风险区间见中文汇总。'
    def chart(name, label, caption):
        return f'''<figure class="main-chart ablation-chart"><a href="assets/zh/{name}.svg" data-zoom data-caption="{label}"><picture><source media="(max-width: 600px)" srcset="assets/zh/{name}-mobile.svg"><img src="assets/zh/{name}.svg" alt="{label}。{caption}" loading="lazy"></picture></a><figcaption>{caption} <a href="assets/zh/{name}.pdf">下载 PDF ↗</a></figcaption></figure>'''
    overview = chart('training-ablation-brier', '完整模型、无事件训练损失、无全局因子的路径概率误差',
                     '青绿为完整模型，橙色为去掉事件损失，紫色为去掉解码器全局因子。柱越短越好；横线表示三次训练的标准差。')
    paired = chart('training-ablation-paired', '六次训练的 Brier 差值与 95% 场景配对区间',
                   '圆点是“消融减完整模型”的误差差值，横线是 95% 场景重采样区间。整段在零右侧支持完整模型更好；跨零表示差异尚不明确。')
    cases = json.loads((SITE / 'data/training_ablation_cases_zh.json').read_text())
    if not cases['passed'] or cases['test_evaluated'] is not False or len(cases['cases']) != 2:
        raise ValueError('two audited fixed visual cases are required')
    examples = []
    for case in cases['cases']:
        panels = []
        for key in visuals['labels']:
            caption = f"{visuals['labels'][key]}：单元格可通行概率图，米白到青绿表示 0 到 1。"
            footprint_caption = f"{visuals['labels'][key]}：第 1 次真实采样，按机器人半径 {case['radius_cells']} 格收缩。绿色允许机器人中心放置，深灰不允许；本次{'有路' if case['first_world_event'][key] else '无路'}。"
            link = zoom(case['panels'][key], caption)
            attributes = f' data-probability-map="{case["panels"][key]}" data-footprint-map="{case["footprint_panels"][key]}" data-probability-caption="{html.escape(caption)}" data-footprint-caption="{html.escape(footprint_caption)}"'
            panels.append(link.replace(' data-zoom ', ' data-zoom' + attributes + ' '))
        panels = ''.join(panels)
        probabilities = '；'.join(f"{visuals['labels'][key]} {case['event_probability'][key]:.1%}" for key in visuals['labels'])
        examples.append(f'''<article class="ablation-case"><h4>参考地图{'有路' if case['target'] else '无路'} · {case['global_id']} · 半径 {case['radius_cells']} 格</h4><p><a href="{case['observed']}" data-zoom data-caption="同一场景的已观测地图，浅灰区域为未知">查看模型输入 ↗</a> · <a href="{case['reference']}" data-zoom data-caption="同一场景的完整参考地图，只用于核对结果">查看完整参考 ↗</a></p><div class="ablation-map-panels">{panels}</div><p class="reading-note">该查询的原验证通路概率：{probabilities}。</p></article>''')
    return f'''<div id="training-ablations" class="training-ablations">
      <p class="small-label">新完成 / 六组训练消融</p><h3>每次去掉一个部分，看它是否有帮助。</h3>
      <p class="ablation-intro">两种消融各独立训练三次，再与三个相同种子的完整模型配对。全部参数训练、精确评估和检查已完成；仍使用相同的验证集。</p>
      {overview}<div class="ablation-conclusions">{conclusions}</div><p class="reading-note"><strong>{risk_note}</strong></p>
      <details id="ablation-details" class="plain-details"><summary>展开消融数值表、差异区间和实验含义</summary><div class="detail-body">
      <div class="table-scroll"><table class="ablation-table"><caption>每种模型三次训练 · 均值 ± 训练种子标准差 · 指标越低越好</caption><thead><tr><th scope="col">模型</th><th scope="col">Brier ↓</th><th scope="col">NLL ↓</th><th scope="col">ECE ↓</th><th scope="col">30% 覆盖率误判 ↓</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
      {paired}<p>去掉事件训练损失仍保留相同的验证事件选模型规则。去掉全局因子仅关闭解码器的全局随机因子，编码器上下文和局部相关性仍在，因此不能解释为“所有空间相关性都被去掉”。</p>
      <p>每个种子内按整个场景重采样 2,000 次；训练波动和场景区间分开报告。验证集曾用于检查点选择，这些区间用于描述当前证据，尚不能替代正式测试。</p>
      <p><a href="https://github.com/s-team-git/ConPath/blob/main/EVALUATION_SUMMARY_ZH.md">中文评估汇总与分层结果 ↗</a> · <a href="data/flatlands_clean_training_ablations.json">完整统计 JSON ↗</a> · <a href="data/training_ablation_metrics.csv">逐次训练 CSV ↗</a> · <a href="data/training_ablation_paired.csv">配对差值 CSV ↗</a></p>
      </div></details>
      <details id="ablation-examples" class="plain-details"><summary>展开真实地图：同一场景，三个模型怎样补全？</summary><div class="detail-body"><p>沿用此前固定的两个验证示例，未按本轮结果重新挑图。三组都展示种子 20260831；手机可左右滑动，点击图片放大。</p><div class="segmented ablation-view" role="group" aria-label="选择消融地图的含义"><button type="button" data-ablation-view="probability" class="selected" aria-pressed="true">每格的平均概率</button><button type="button" data-ablation-view="footprint" aria-pressed="false">一次采样 + 机器人尺寸</button></div><p id="ablation-view-note" class="reading-note">每幅图使用相同的 0–1 色条。颜色都很绿，也不保证整条通路能同时容纳机器人；切换右侧视图查看一次实际采样。</p>{''.join(examples)}<p class="provenance">图片来自真实检查点，以固定绘图种子采样 128 次、每批 8 张。“一次采样”固定展示第 1 张完整世界，经圆盘半径收缩后的机器人中心可放置区域，未按成败挑选。图中通路概率来自原精确验证 CSV，与绘图使用不同随机流；0% 表示 128 次中未采到有路，不是绝对不可能的证明。两个例子用于解释，不能代替整体统计。S 为起点、G 为目标，没有绘制规划路线。<a href="data/training_ablation_cases_zh.json">模型、图片与查询来源 ↗</a></p></div></details></div>'''


def cogniplan_native_sanity():
    path = SITE / 'data/cogniplan_native_sanity_zh.json'
    if not path.is_file():
        return ''
    data = json.loads(path.read_text())
    if data['formal_comparison'] or data['test_evaluated'] or len(data['cases']) != 3:
        raise ValueError('Native sanity must remain separate from formal comparisons')
    for asset in data['assets']:
        if hashlib.sha256((SITE / asset['path']).read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError(f'Native image hash mismatch: {asset["path"]}')
    cases = []
    for case in data['cases']:
        panels = ''.join(zoom(case['panels'][key], caption) for key, caption in
                         [('observed', '模型输入：浅绿可通行，深灰阻挡，浅灰未知。'),
                          ('vote', '官方模型四张补全的可通行比例：米白到青绿对应 0 到 1；这不是经过校准的通路概率。'),
                          ('reference', '完整参考地图，仅用于检查输出，不输入推理模型。')])
        worlds = ''.join(zoom(case['panels'][f'world{k}'], f'固定条件{k+1}：{name}。四个条件均运行，未用真实布局选择。')
                         for k, name in enumerate(['均衡', '房间', '隧道', '户外']))
        cases.append(f'''<details class="plain-details native-case"><summary>{case['layout_zh']}地图 · {case['mother_id']} · 查看真实输出</summary><div class="detail-body"><div class="native-map-panels">{panels}</div><details class="plain-details"><summary>查看四张原生补全地图</summary><div class="native-map-panels">{worlds}</div></details></div></details>''')
    return f'''<div id="external-progress" class="training-ablations"><p class="eyebrow">外部对比准备 · 2026.09.08</p><h3>CogniPlan 已接通，先核对真实输出。</h3><p>已在32个固定训练观测上生成128张地图，与官方后处理逐张一致，已观测区域零冲突。重建和对抗阶段各完成100次更新测速，未启动正式长训练。</p><p class="reading-note"><strong>这些是原训练数据上的接口检查，不是留出验证成绩。</strong>公开权重见过原训练地图；公平主表将按新的共同划分重新训练。四个输出来自预设条件，未用真实布局选取。以下每类首例在看结果前已经固定，输出不准确的部分也保留。</p>{''.join(cases)}<p>新划分：2,400张母地图用于训练、300张校准、300张验证；旋转和镜像后的重复几何也检查过。初步测速外推，原版50万步约需{data['training_hours_per_seed_extrapolation']:.1f}小时/次，三次串行约{data['training_hours_three_seeds_extrapolation']:.1f}小时，另加完整数据读取、评估与存档时间。当前有其他GPU负载，此处只用于估算排期。</p><p class="source-line"><a href="https://github.com/s-team-git/ConPath/blob/main/EXTERNAL_PROGRESS_ZH.md">接口、尺度与测速中文记录 ↗</a> · <a href="data/cogniplan_native_sanity_zh.json">图片与运行来源 ↗</a></p></div>'''


def flatlands_external_engineering():
    path = SITE / 'data/flatlands_external_engineering_zh.json'
    if not path.is_file():
        return ''
    data = json.loads(path.read_text())
    if data['formal_comparison'] or data['test_evaluated'] or not data['engineering_only']:
        raise ValueError('External engineering snapshots must not become formal results')
    for asset in data['assets']:
        if hashlib.sha256((SITE / asset['path']).read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError(f'External image hash mismatch: {asset["path"]}')
    rows = []
    for key, name in [('lama', 'LaMa：Fourier卷积补全'), ('flow', '条件流匹配＋交叉注意力')]:
        p, b = data['profiles'][key], data['batch64'][key]
        if p['status'] != 'complete' or b['optimizer_updates_measured'] != 100:
            raise ValueError('Both completed native-size profiles are required')
        rows.append(f'<tr><th scope="row">{name}</th><td>{p["generator_parameters"]/1e6:.2f} 百万</td><td>{p["inference_actual_cost_per_observation"]["samples_per_observation"]}张</td><td>{b["seconds_mean"]:.2f}秒</td><td>{b["peak_allocated_bytes"]/2**30:.2f} GiB</td></tr>')
    cases = []
    for case in data['cases']:
        panels = ''.join(zoom(case['panels'][key], caption) for key, caption in
                        [('observed', '模型输入：浅绿可通行，深灰阻挡，浅灰未知，灰米色为有效范围外。'),
                         ('lama', 'LaMa短训练输出值，0到1使用米白至青绿；尚未收敛，不是通路概率。'),
                         ('flow', '流匹配四张图的可通行比例，0到1使用米白至青绿；尚未收敛，不是通路概率。'),
                         ('reference', '完整参考地图，仅用于核对，推理模型不会读取。')])
        worlds = ''.join(zoom(case['panels'][f'flow_world{k}'], f'流匹配第{k+1}张完整采样；未按答案选择，短训练尚未收敛。') for k in range(4))
        cases.append(f'<details class="plain-details external-case"><summary>{case["source"]} · {case["global_id"]} · 查看短训练输出</summary><div class="detail-body"><div class="external-map-panels">{panels}</div><details class="plain-details"><summary>四张实际生成的完整地图</summary><div class="external-map-panels">{worlds}</div></details></div></details>')
    return f'''<div id="flatlands-external-progress" class="training-ablations"><p class="eyebrow">外部方法接入 · 2026.09.08</p><h3>LaMa 与条件流匹配已完成首轮实测。</h3><p>保留LaMa的官方Fourier骨干，流匹配按论文方程重实现；两者均完成100次有效批量64的更新测速。所有模型接收相同观测与有效范围，输出恢复已观测区域并封闭范围外。</p><div class="table-scroll"><table class="literature-table external-cost-table"><thead><tr><th>方法</th><th>生成器参数量</th><th>每次输出</th><th>批量64每次更新</th><th>训练峰值显存</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="reading-note">当前GPU存在其他任务，这张表用于排期。更新次数不等于轮数；梯度累积、原始耗时和输入传输范围见中文记录。LaMa这一行只跑了单模型，四成员集成尚未训练。流匹配用25步Heun生成四张图，实测50次批量前向，计入条件引导相当于400次单图前向。</p><p><strong>以下是固定训练样本上的接口检查，模型尚未收敛。</strong>观测、模型输出、完整参考分别标注；错误和不合理的补全也保留。没有把这些短训练分数放入正式对比表。</p>{''.join(cases)}<details class="plain-details"><summary>查看本轮训练过程与限制</summary><div class="detail-body"><p>图片来自首轮小批量短训练；以下曲线来自另一次有效批量64的测速。32个训练观测被反复使用，损失下降不证明验证有效，也不能据此判断哪个方法更好。</p><div class="external-training-curves">{zoom('assets/zh/flatlands-external/lama-training-loss.svg', 'LaMa有效批量64的重建损失，仅训练数据。')}{zoom('assets/zh/flatlands-external/flow-training-loss.svg', '条件流匹配有效批量64的速度MSE，仅训练数据。')}</div><p class="source-line">下载训练曲线：<a href="assets/zh/flatlands-external/lama-training-loss.pdf">LaMa PDF ↗</a> · <a href="assets/zh/flatlands-external/flow-training-loss.pdf">流匹配 PDF ↗</a></p></div></details><p class="source-line"><a href="https://github.com/s-team-git/ConPath/blob/main/FLATLANDS_EXTERNAL_PROGRESS_ZH.md">本轮实现、成本与下一步 ↗</a> · <a href="data/flatlands_external_engineering_zh.json">配置、图片与来源 ↗</a></p></div>'''


def main():
    data=json.loads((SITE/'data/site_visuals_zh.json').read_text())
    analysis=json.loads((SITE/'data/flatlands_clean_paper_analysis.json').read_text())
    ablations = training_ablations()
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
      <p class="status-line"><span class="status-dot"></span>验证阶段 · 新消融评估已恢复 · 更新于 2026.09.08</p>
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
      {{ABLATIONS}}
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
      <details class="plain-details"><summary>主实验具体比较什么？</summary><div class="detail-body"><p>第一张新表计划统一为 4 个完整地图输出，比较 ConPath、LaMa 集成、条件流匹配等；确定性方法保留单个输出。另一张表记录实际采样量、误差、耗时和显存。现有 128 次采样的数值不会直接混进这张新表。</p><p>实验方案已确认：补齐外部方法对比、三次独立训练、场景配对统计和失败案例，再写入论文。已知限制也会保留：均值地图结果很接近，外部强基线尚未运行，FlatLands 原文与本地元数据的尺度差异待核对。六组消融已按要求恢复，先完成精确评估，再接续后面的训练；当前没有新增的最终结果。</p></div></details>
      <p class="source-line"><a href="https://github.com/s-team-git/ConPath/blob/main/LITERATURE_REVIEW_ZH.md">八篇论文怎样做比较 ↗</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/EXPERIMENT_DESIGN_ZH.md">具体对比实验方案 ↗</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/DATASET_CHOICE_ZH.md">数据集选型分析 ↗</a></p>
    </div></section>
    <section class="section shell closing"><h2>目前做到哪一步？</h2><p>主验证结果与图表已完成。实验已按要求恢复：三组训练分别完成 12、9、17 轮，现在重新进行未完成的精确评估；另外三组自动排队。全部完成并核验后更新表格与图片。<br>论文仍是工作草稿；已确认的后续工作包括外部方法主对比、CogniPlan 原生地图实验、计算成本和统计分析。当前页面展示已完成的验证结果。</p><div class="resource-links"><a href="https://github.com/s-team-git/ConPath/blob/main/WORK_PLAN.md">中文工作计划 ↗</a><a href="https://github.com/s-team-git/ConPath/blob/main/EXPERIMENT_DESIGN_ZH.md">已确认的实验方案 ↗</a><a href="https://github.com/s-team-git/ConPath/blob/main/CLEAN_ABLATION_PLAN.md">消融方案 ↗</a><a href="archive/index.html">旧版页面与图表归档 ↗</a></div></section>
  </main>
  <footer class="shell footer"><p>ConPath · 可复现研究记录</p><p>页面结构参考 <a href="https://s-team-git.github.io/Lightweight-3DGS/">Lightweight-3DGS</a>，重新实现中文排版与交互。<br>FlatLands 派生地图保留各上游来源条款；UnScenes3D 数据按其 CC BY 4.0 使用说明署名。<a href="https://github.com/s-team-git/ConPath/blob/main/DATASET_CHOICE_ZH.md">数据来源与使用说明</a>。</p></footer>
  <dialog id="image-dialog" aria-labelledby="dialog-caption"><div class="dialog-toolbar"><p id="dialog-caption"></p><button type="button" id="dialog-close" aria-label="关闭放大图片">关闭 ×</button></div><div class="dialog-image"><img id="dialog-image" alt="放大后的当前图片"></div><a id="dialog-source" href="#top">打开原始尺寸 ↗</a></dialog>
  <p id="interaction-error" class="interaction-error" hidden>交互数据暂时加载失败；已展示首个示例与完整静态结果，请刷新重试。</p>
</body></html>
'''
    for key,value in [('PANELS',panels),('GALLERY',gallery),('GALLERY_SOURCE',gallery_source),('THUMBNAILS',thumbs),('SEQUENCE',sequence),('ROWS','\n'.join(rows)),('ABLATIONS',ablations)]:page=page.replace('{{'+key+'}}',value)
    if ablations:
        page = page.replace('新消融评估已恢复', '六组训练消融已完成')
        page = page.replace('六组消融已按要求恢复，先完成精确评估，再接续后面的训练；当前没有新增的最终结果。', '六组内部训练消融及配对统计已完成，结果见上方新表。接下来补齐外部方法主比较；内部消融不能代替这些对比实验。')
        page = page.replace('主验证结果与图表已完成。实验已按要求恢复：三组训练分别完成 12、9、17 轮，现在重新进行未完成的精确评估；另外三组自动排队。全部完成并核验后更新表格与图片。', '本轮六组参数训练与精确评估均已完成。中文汇总包含三次训练结果、场景配对区间和按来源/半径的分层表现，图表与论文同步更新。')
        page = page.replace('<div class="publication-links">', '<div class="publication-links"><a class="pill" href="https://github.com/s-team-git/ConPath/blob/main/EVALUATION_SUMMARY_ZH.md">中文评估汇总 ↗</a>')
    external = cogniplan_native_sanity()
    if external:
        page = page.replace('下面的新增对照仍是计划，尚未运行。', '正式外部主比较尚未运行；CogniPlan 已完成原生接口检查和小规模测速，记录如下。')
        page = page.replace('<p class="research-decision">', external + '<p class="research-decision">')
        page = page.replace('六组训练消融已完成 · 更新于', '六组消融完成 · 外部模型接口已接通 · 更新于')
        page = page.replace('外部强基线尚未运行，FlatLands 原文与本地元数据的尺度差异待核对。', '正式外部主比较尚未运行。FlatLands 的160份训练元数据均描述裁剪，但论文另有缩放描述；物理尺度仍未独立验证，半径继续按格报告。')
        page = page.replace('当前页面展示已完成的验证结果。', '当前页面分别展示已完成的验证结果和 CogniPlan 训练数据接口检查；下一项是 LaMa/流匹配接入及正式训练配方冻结。')
    external_flatlands = flatlands_external_engineering()
    if external_flatlands:
        page = page.replace('<p class="research-decision">', external_flatlands + '<p class="research-decision">')
        page = page.replace('下一项是 LaMa/流匹配接入及正式训练配方冻结。', 'LaMa与流匹配已完成接入和测速，下一项是统一正式数据规模、查询规则与收敛诊断。')
        page = page.replace('CogniPlan 已完成原生接口检查和小规模测速，记录如下。', 'CogniPlan、LaMa和流匹配已完成接口检查和小规模测速，记录如下。')
    assert '{{' not in page
    css_version = hashlib.sha256((SITE / 'styles-zh.css').read_bytes()).hexdigest()[:12]
    page = page.replace('href="styles-zh.css"', f'href="styles-zh.css?v={css_version}"')
    (SITE/'index.html').write_text(page)
    print('Chinese page built: 9 methods, 2 model examples, 12 gallery scenes, 18 frames.')


if __name__=='__main__':main()
