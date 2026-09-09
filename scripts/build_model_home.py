#!/usr/bin/env python3
"""An effects-first homepage from published, unchanged model-output bitmaps.

This page builder opens no dataset/archive/checkpoint; it includes separately audited new inference. SVG text labels are simplified; the
embedded map bitmap and every S/G coordinate remain exactly as published.
"""
import copy
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
NS = '{http://www.w3.org/2000/svg}'
ET.register_namespace('', 'http://www.w3.org/2000/svg')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_card(key, label, path, caption):
    return f'''<figure class="output-card" data-panel="{key}"><figcaption><span>{label}</span><p>{caption}</p></figcaption><a href="{path}" data-image data-caption="{html.escape(label+'：'+caption)}"><img id="image-{key}" src="{path}" alt="{html.escape(label+'：'+caption)}" loading="eager" decoding="async"><span class="image-zoom">点击放大 ↗</span></a></figure>'''


def main():
    original = json.loads((SITE/'data/site_visuals_zh.json').read_text())
    stats = json.loads((SITE/'data/current_baseline_k4_analysis.json').read_text())
    scope = json.loads((SITE/'data/flatlands_read_scope_erratum.json').read_text())
    assert scope['historical_physical_test_untouched_claim_withdrawn']
    folder = SITE/'assets/zh/model-examples'; folder.mkdir(parents=True, exist_ok=True)
    cases = copy.deepcopy(original['examples'])
    copied = []
    for case in cases:
        for key, relative in list(case['panels'].items()):
            source = SITE/relative
            assert sha(source) == original['assets'][relative]['sha256']
            tree = ET.fromstring(source.read_text())
            bitmaps = [n.attrib.copy() for n in tree.findall(f'{NS}image')]
            markers = [ET.tostring(n) for n in tree if n.tag in (f'{NS}circle', f'{NS}path')]
            name = 'ConPath' if key.startswith('correlated') else '独立对照'
            label = {'observed': '机器人已看到的地图', 'reference': '真实参考地图'}.get(
                key, name + (' · 一次实际补全' if key.endswith('_sample') else ' · 每格可通行概率'))
            tree.find(f'{NS}title').text = label
            heading = next(n for n in tree.findall(f'{NS}text') if n.get('y') == '26')
            heading.text = label
            destination = folder/f'{case["id"]}-{key}.svg'
            destination.write_text(ET.tostring(tree, encoding='unicode')+'\n')
            verify = ET.fromstring(destination.read_text())
            assert [n.attrib for n in verify.findall(f'{NS}image')] == bitmaps
            assert [ET.tostring(n) for n in verify if n.tag in (f'{NS}circle', f'{NS}path')] == markers
            case['panels'][key] = str(destination.relative_to(SITE))
            copied.append({'path': case['panels'][key], 'sha256': sha(destination), 'source': relative,
                           'source_sha256': sha(source), 'embedded_map_and_endpoint_geometry_unchanged': True})
    expanded = json.loads((SITE/'data/expanded_model_gallery_zh.json').read_text())
    for case in cases:
        case.update(domain='indoor', samples=128, split_zh='旧解释性案例', checkpoint_scope_zh='室内训练检查点；此前按标签选定', selection='historical_label_selected')
    cases = expanded['examples'] + cases
    copied += expanded['assets']
    pilot_status = json.loads((SITE/'data/parent_group_pilot_status_zh.json').read_text()) if (SITE/'data/parent_group_pilot_status_zh.json').exists() else None
    home_data = {'examples': cases, 'baseline_training_snapshot': pilot_status, 'model_training_samples': 160, 'example_samples': [32, 128],
                 'latest_evaluation_samples': 4, 'new_training_for_gallery': False,
                 'new_dataset_images_opened': True, 'new_physical_test_images_opened': 0, 'new_cases': expanded['new_cases'], 'historical_model_pixels_changed': False,
                 'selection_note': '28 new metadata-selected development cases, plus 2 preserved historical label-selected examples. Each shows the first actual world; none is best-of-K.',
                 'examples_are_old_cohort_diagnostics': True, 'assets': copied,
                 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [SITE/'data/site_visuals_zh.json', SITE/'data/expanded_model_gallery_zh.json', SITE/'data/current_baseline_k4_analysis.json', Path(__file__)]}}
    data_path = SITE/'data/model_home_zh.json'
    data_path.write_text(json.dumps(home_data, ensure_ascii=False, indent=2)+'\n')
    pilot_note = f'父级地点隔离的小规模基线正在进行：{pilot_status["completed"]}/9次训练完成（{pilot_status["timestamp_utc"][:16]} UTC快照）。图集仍展示此前的检查点。' if pilot_status else '小规模基线正在准备。'
    pilot_paused = bool(pilot_status and pilot_status['stage'] == 'paused_by_user')
    if pilot_paused:
        pilot_note = f'已按用户要求暂停：{pilot_status["completed"]}/9次训练完成（{pilot_status["timestamp_utc"][:16]} UTC快照）。其余训练保留检查点，尚未统一验证评分。'
    training_heading = '训练已暂停，检查点已保存' if pilot_paused else '正在从头训练三种模型'
    training_description = '九次训练中3次完成，另外两次保存到第12轮和第11轮；等待明确恢复指令。' if pilot_paused else 'ConPath、独立模型、确定性网络各三个随机种子，最多24轮；全部选优结束后统一评分，并比较简单规则。'
    case = cases[0]
    cards = ''.join(image_card(key, label, case['panels'][source], caption) for key, label, source, caption in [
        ('input', '① 模型输入', 'observed', '只有已经观测到的部分；浅灰区域未知。'),
        ('prediction', '② 模型预测', 'correlated_sample', '一次实际采样的完整地图。'),
        ('reference', '③ 真实参考', 'reference', '完整地图，用于核对；不输入推理模型。')])
    thumbnails = ''.join(f'<button data-case="{i}" aria-pressed="{str(i == 0).lower()}" class="case-thumb{" selected" if i == 0 else ""}"><img src="{row["panels"]["correlated_sample"]}" alt="ConPath补全预览 {i+1}：{row["source"]}" loading="lazy"><span>{row["source"]} · {i+1:02d}</span></button>' for i, row in enumerate(cases))
    sources = ''.join(f'<option value="{name}">{name}（{sum(c["source"] == name for c in cases)}组）</option>' for name in sorted({c['source'] for c in cases}))
    labels = {'correlated':'ConPath', 'independent':'独立单元模型', 'tiny_deterministic':'确定性补全网络', 'all_floor':'未知全部可通行'}
    rows = []
    for key, name in labels.items():
        m = stats['methods'][key]
        rows.append(f'<tr data-home-method="{key}"><th scope="row">{name}</th><td>{m["samples"]}</td><td>{m["brier"]["mean"]:.4f}</td><td>{m["risk30"]["mean"]*100:.2f}%</td></tr>')
    css = sha(SITE/'home.css')[:12]; js = sha(SITE/'home.js')[:12]
    embedded_data = json.dumps(home_data, ensure_ascii=False).replace('<', '\\u003c')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f8faf9"><title>ConPath · 模型效果与实验结果</title><meta name="description" content="直接查看ConPath真实模型输出：输入、预测与参考地图并排比较，保留失败例子。"><link rel="icon" href="favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="home.css?v={css}"><script src="home.js?v={js}" defer></script></head>
<body><a class="skip" href="#effects">跳到模型效果</a><header class="header"><nav class="wrap navigation" aria-label="主导航"><a class="brand" href="index.html">ConPath<span>研究项目</span></a><div><a href="#effects">模型效果</a><a href="#results">实验结果</a><a href="#next">下一步</a><a href="research.html">研究记录 ↗</a></div></nav></header>
<main class="wrap"><section class="intro"><p class="eyebrow">部分地图 → 完整世界 → 通路概率</p><h1>模型补全得怎么样？<br><span>把预测和真实地图放在一起看。</span></h1><p>ConPath根据已观测区域推测完整地图，再结合机器人尺寸估计两个点之间是否有路。</p><p class="status">新增28组真实输出：20组室内、8组室外；加上原有2组，共30组可切换查看。</p><p class="status">{pilot_note}</p></section>
<section id="effects" class="section effects"><span id="method" class="anchor-alias"></span><div class="section-heading"><div><p class="eyebrow">01 / 真实模型输出</p><h2>输入、预测、参考</h2></div><span id="case-badge" class="badge">开发验证 · 30组案例</span></div>
<div class="controls"><label class="model-select">数据来源<select id="home-source"><option value="all">全部（30组）</option>{sources}</select></label><div class="case-paging"><button id="case-prev" aria-label="上一个案例">←</button><span id="case-counter">1 / 30</span><button id="case-next" aria-label="下一个案例">→</button></div><label class="model-select">模型<select id="home-model"><option value="correlated">ConPath</option><option value="independent">独立单元对照</option></select></label></div>
<div class="case-strip" aria-label="选择ConPath效果案例">{thumbnails}</div><div class="view-row"><div class="segmented light" aria-label="模型输出显示方式"><button data-view="sample" class="selected" aria-pressed="true">一次实际补全</button><button data-view="probability" aria-pressed="false">每格概率图</button></div><p id="view-explanation">绿色表示可通行，深灰表示阻挡。</p></div>
<div class="mobile-panels" aria-label="切换对照图片"><button data-jump="0" class="selected" aria-pressed="true">① 输入</button><button data-jump="1" aria-pressed="false">② 预测</button><button data-jump="2" aria-pressed="false">③ 参考</button></div>
<div id="output-gallery" class="output-gallery">{cards}</div>
<div class="legend" aria-label="地图图例"><span><i class="free"></i>可通行</span><span><i class="blocked"></i>阻挡</span><span><i class="unknown"></i>未知</span><span><i class="outside"></i>有效范围外</span><span><b class="start">● S</b> 起点</span><span><b class="goal">◆ G</b> 目标</span></div>
<div id="case-reading" class="case-reading" aria-live="polite"><div><span>模型估计有路概率</span><strong id="case-probability">{case["event_probability"]["correlated"]*100:.1f}%</strong></div><p id="case-explanation">固定显示第一次实际补全；请对照右侧真实参考查看差异。</p></div>
<p id="case-source" class="caption">{case["source"]} · {case["global_id"]} · 机器人半径10格 · 通路概率来自同一批32次实际采样。</p>
<p class="caption">S、G只是查询端点，图中没有绘制规划路线。新增28组按来源、编号和时间固定选择，未按模型效果筛选；最后两组保留此前按标签选定的解释性案例。所有案例都来自开发阶段，不能代替独立最终测试统计。<a href="research.html#method">查看完整来源与原示例 ↗</a></p>
<aside class="evidence-note"><strong>结果适用范围</strong><p>新增室内案例排除了与旧训练共享地点的观测，但仍曾参与旧模型选优；室外使用单独训练的模型，并依赖数据集提供的有效区域。图片和旧表格用于诊断，尚不能证明最终测试泛化或超过其它论文。<a href="research.html#baseline-review">查看审计与更正 ↗</a></p></aside>
</section>
<section id="results" class="section"><div class="section-heading"><div><p class="eyebrow">02 / 既有整体评估</p><h2>有改善，也有明显的不足。</h2></div></div><p class="section-description">下表保留旧队列结果；新基线完成后另表报告，不混用不同划分。下表使用同一批查询。随机方法输出4张地图，确定性方法输出1张；这与上方128次采样的解释性示例分开报告。</p>
<div class="table-scroll"><table><caption>三次训练的均值；无训练规则仅一次结果。两项指标均越低越好。</caption><thead><tr><th scope="col">方法</th><th scope="col">输出数</th><th scope="col">概率误差 ↓</th><th scope="col">30%覆盖率误判 ↓</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="result-reading">ConPath的概率误差低于独立模型，但“全部可通行”的简单规则更低；该规则的误判风险又更高。当前没有全面领先的证据。</p><p class="caption">概率误差指Brier；30%覆盖率误判指只接受评分最高的30%查询后，实际无路的比例。<a href="research.html#baseline-review">完整6种方法、波动区间与图表 ↗</a></p></section>
<section id="next" class="section"><div class="section-heading"><div><p class="eyebrow">03 / 下一步</p><h2>先确认优势在干净数据上是否成立。</h2></div></div><ol class="next-steps"><li><span>01</span><div><h3>小规模数据检查已通过</h3><p>100个训练、25个选优、40个验证地点，地点隔离与地图重复检查通过。查询保留目标落在障碍上的自然负例。</p></div></li><li><span>02</span><div><h3>{training_heading}</h3><p>{training_description}</p></div></li><li><span>03</span><div><h3>再验证室外历史融合</h3><p>先修正观测置信度与地面表达，再测试历史几何融合，最后加入注意力；分别检验数据和架构的贡献。</p></div></li></ol><p class="caption">{pilot_note} <a href="https://github.com/s-team-git/ConPath/blob/main/PARENT_GROUP_PILOT_ZH.md">本轮固定协议 ↗</a>。外部对比优先复用可比较的论文成绩、作者预测或权重。<a href="https://github.com/s-team-git/ConPath/blob/main/WORK_PLAN.md">完整工作计划 ↗</a></p></section>
<section class="records-links" aria-label="深入阅读"><a href="research.html#datasets"><strong>数据集与实景</strong><span>训练数据示例、相机与俯视地图 ↗</span></a><a href="research.html#training-ablations"><strong>内部消融</strong><span>完整模型与两个删减版本 ↗</span></a><a href="research.html#flatlands-external-progress"><strong>外部方法检查</strong><span>短训练输出，尚不是正式对比 ↗</span></a></section>
</main><footer class="wrap footer"><span>ConPath · 可复现研究记录</span><div><a href="research.html">全部研究记录</a><a href="https://github.com/s-team-git/ConPath/blob/main/BASELINE_REVIEW_ZH.md">中文分析</a><a href="https://github.com/s-team-git/ConPath">GitHub</a></div><p>FlatLands派生数据保留上游来源条款。<a href="https://github.com/s-team-git/ConPath/blob/main/DATASET_CHOICE_ZH.md">数据来源与使用说明</a></p></footer>
<dialog id="model-dialog" aria-labelledby="model-dialog-caption"><div class="dialog-toolbar"><p id="model-dialog-caption"></p><button id="model-dialog-close" aria-label="关闭放大图片">关闭 ×</button></div><img id="model-dialog-image" alt="当前模型图片的放大视图"><a id="model-dialog-source" href="#effects">打开原始图片 ↗</a></dialog><p id="home-error" hidden>交互加载失败，仍可查看当前静态示例；请刷新重试。</p>
<script type="application/json" id="home-data">{embedded_data}</script></body></html>'''
    (SITE/'index.html').write_text(page+'\n')
    print(f'Homepage: {len(cases)} real model cases, 3 comparison panels, 4 summary rows.')


if __name__ == '__main__':
    main()
