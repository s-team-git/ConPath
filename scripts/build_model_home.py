#!/usr/bin/env python3
"""An effects-first homepage from published, unchanged model-output bitmaps.

This page builder opens no dataset/archive/checkpoint; it includes separately audited new inference. SVG text labels are simplified; the
embedded map bitmap and every S/G coordinate remain exactly as published.
"""
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
NS = '{http://www.w3.org/2000/svg}'
ET.register_namespace('', 'http://www.w3.org/2000/svg')
PILOT_METHODS = {'correlated': 'ConPath', 'independent': '独立单元模型',
                 'deterministic': '确定性补全网络', 'all_floor': '未知全部可通行',
                 'train_radius_prior': '训练集半径先验'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_card(key, label, path, caption):
    return f'''<figure class="output-card" data-panel="{key}"><figcaption><span>{label}</span><p>{caption}</p></figcaption><a href="{path}" data-image data-caption="{html.escape(label+'：'+caption)}"><img id="image-{key}" src="{path}" alt="{html.escape(label+'：'+caption)}" loading="eager" decoding="async"><span class="image-zoom">点击放大 ↗</span></a></figure>'''


def verified_pilot():
    """Read only published metadata and map assets; incomplete reports stay off home."""
    paths = [SITE/'data/parent_pilot_gallery_zh.json', SITE/'data/parent_group_pilot_zh.json',
             SITE/'data/parent_group_pilot_verification.json', SITE/'pilot.html',
             SITE/'data/parent_pilot_gallery_verification.json']
    if not all(path.is_file() for path in paths):
        return None, 'awaiting_complete_published_report'
    try:
        gallery, result, verification = [json.loads(path.read_text()) for path in paths[:3]]
        gallery_verification = json.loads(paths[4].read_text())
        checksum = sha(paths[1])
        assert verification['passed'] is True
        assert verification['analysis_sha256'] == gallery['source_analysis_sha256'] == checksum
        assert gallery_verification['passed'] is True
        assert gallery_verification['analysis_sha256'] == checksum
        assert gallery_verification['gallery_sha256'] == sha(paths[0])
        assert result['id'] == 'parent_group_pilot_v1'
        assert result['implementation_checks_passed'] is True and result['training_runs_completed'] == 9
        assert result['final_test'] is False and result['direct_comparison_to_published_scores'] is False
        assert isinstance(result['research_screen_passed'], bool)
        assert result['validation_parents'] > 0 and result['validation_events'] > 0
        for method in PILOT_METHODS:
            entry = result['methods'][method]
            assert entry['repeats'] == (3 if method in ('correlated', 'independent', 'deterministic') else 1)
            budget = '32' if method in ('correlated', 'independent') else '1'
            for metric in ('brier', 'risk30'):
                value = entry['budgets'][budget][metric]['mean']
                assert isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
        cases = gallery['examples']
        assert len(cases) == 10 and len({row['id'] for row in cases}) == len(cases)
        assets = {asset['path']: asset['sha256'] for asset in gallery['assets']}
        for row in cases:
            assert row['cohort'] == 'new_pilot' and row['final_test'] is False
            assert row['checkpoint_selected_on_this_cohort'] is False and row['samples'] == 32
            assert isinstance(row['target'], bool) and row['radius_cells'] >= 0
            assert all(row[key] for key in ('source', 'global_id', 'split_zh', 'checkpoint_scope_zh'))
            for model in ('correlated', 'independent'):
                value = row['event_probability'][model]
                assert isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                assert isinstance(row['displayed_world_event'][model], bool)
            for key in ('observed', 'reference', 'correlated', 'correlated_sample', 'independent', 'independent_sample'):
                relative = row['panels'][key]
                asset = (SITE/relative).resolve()
                assert asset.is_relative_to(SITE.resolve()) and asset.is_file()
                assert sha(asset) == assets[relative]
        return {'gallery': gallery, 'result': result, 'source_paths': paths}, 'verified'
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f'Pilot report withheld: published verification or asset checks failed ({type(error).__name__}).')
        return None, 'published_report_validation_failed'


def training_copy(status, pilot_ready, candidate=None):
    """Describe the recorded state without turning a dated snapshot into live status."""
    if pilot_ready:
        note = '父级地点隔离的小规模基线已完成并通过结果核验；新案例与本轮评估已更新，仍属于开发验证，非最终测试。'
        heading = '三种模型的基线训练与核验已完成'
        description = '三种模型各三个随机种子；本轮结果已单独报告，下一项改进依据这批开发验证的具体差距确定。'
        if candidate:
            snapshot = candidate['timestamp_utc'][:16].replace('T', ' ') + ' UTC快照'
            if candidate['stage'] == 'training':
                heading = '两个种子的连续采样改进正在训练'
                description = '只改变最后一层随机采样的空间相关性，训练预算保持不变；两个种子完成后，与原模型的相同两个种子比较。'
            elif candidate['stage'] == 'paused_by_user':
                heading = '改进试验已按用户要求暂停'
                description = '已有基线和效果图保留；改进试验从已保存的完整轮次恢复。'
            elif candidate['stage'] in ('complete', 'evaluating'):
                heading = '两种子改进训练已结束，正在整理比较'
                description = '新结果通过核验并完整发布后报告；不提前用选优集成绩判断改进成功。'
            else:
                heading = '两种子改进试验等待检查'
                description = '以已发布的结果和状态为准；不自动扩大规模。'
            note += f' 改进试验已完成{candidate["completed"]}/2次训练（{snapshot}，非实时监控）。'
    elif status:
        progress = f'{status["completed"]}/{status["total"]}次训练完成'
        snapshot = f'（{status["timestamp_utc"][:16].replace("T", " ")} UTC快照，非实时监控）'
        stage = status['stage']
        if stage == 'paused_by_user':
            note = f'已按用户要求暂停：{progress}{snapshot}。图集仍展示此前的检查点。'
            heading = '训练已暂停，检查点已保存'
            description = f'{progress}；其余进度已保存，尚未完成统一验证评分。'
        elif stage == 'baseline_complete':
            note = f'本轮基线训练与评分已完成：{progress}{snapshot}。完整报告核验与发布尚未完成，当前展示旧案例和旧队列结果。'
            heading = '基线评分已完成，等待报告核验与发布'
            description = '新结果通过独立核验并完整发布后，首页才会更新本轮案例和结果表。'
        elif stage == 'training':
            note = f'父级地点隔离的小规模基线正在进行：{progress}{snapshot}。图集仍展示此前的检查点。'
            heading = '正在从头训练三种模型'
            description = f'ConPath、独立模型、确定性网络各三个随机种子，最多{status.get("max_epochs", 24)}轮；全部选优结束后统一评分，并比较简单规则。'
        else:
            note = f'本轮训练记录：{progress}{snapshot}。新报告尚未通过发布核验，当前展示旧案例和旧队列结果。'
            heading = '等待下一份训练与评分记录'
            description = '以已发布的状态快照为准；本轮案例和结果通过核验后更新。'
    else:
        note = '本轮基线尚无已发布的状态快照；当前展示此前的模型案例与旧队列结果。'
        heading = '等待本轮基线状态记录'
        description = '三种模型各计划三个随机种子，完成选优和统一评分后核验结果。'
    policy = status.get('resource_policy') if status else None
    resource_note = ''
    if isinstance(policy, dict):
        parts = []
        if policy.get('max_training_processes') is not None:
            parts.append(f'同时最多{policy["max_training_processes"]}个训练进程')
        if policy.get('project_gpu_usage_target_gib') is not None:
            parts.append(f'本项目显存目标约{policy["project_gpu_usage_target_gib"]:g} GiB')
        if policy.get('allocator_gib_per_worker') is not None:
            parts.append(f'每进程分配器上限{policy["allocator_gib_per_worker"]:g} GiB')
        if parts:
            resource_note = '训练资源策略：' + '，'.join(parts) + '。这是配置目标，不是实时占用；其它程序的显存另计。'
    elif isinstance(policy, str) and policy:
        resource_note = f'训练资源策略：{policy}。这是配置目标，不是实时占用；其它程序的显存另计。'
    return note, heading, description, resource_note


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
    for case in cases:
        case['cohort'] = 'historical'
    copied += expanded['assets']
    old_count = len(cases)
    pilot, pilot_state = verified_pilot()
    new_count = len(pilot['gallery']['examples']) if pilot else 0
    if pilot:
        cases = pilot['gallery']['examples'] + cases
        copied += pilot['gallery']['assets']
    pilot_status = json.loads((SITE/'data/parent_group_pilot_status_zh.json').read_text()) if (SITE/'data/parent_group_pilot_status_zh.json').exists() else None
    candidate_status = json.loads((SITE/'data/coherent_pilot_status_zh.json').read_text()) if (SITE/'data/coherent_pilot_status_zh.json').exists() else None
    source_paths = [SITE/'data/site_visuals_zh.json', SITE/'data/expanded_model_gallery_zh.json', SITE/'data/current_baseline_k4_analysis.json', Path(__file__)]
    if pilot:
        source_paths += pilot['source_paths']
    if pilot_status:
        source_paths.append(SITE/'data/parent_group_pilot_status_zh.json')
    if candidate_status:
        source_paths.append(SITE/'data/coherent_pilot_status_zh.json')
    result_methods = list(PILOT_METHODS) if pilot else ['correlated', 'independent', 'tiny_deterministic', 'all_floor']
    home_data = {'examples': cases, 'baseline_training_snapshot': pilot_status, 'candidate_training_snapshot': candidate_status, 'historical_model_training_samples': 160,
                 'example_samples': sorted({row['samples'] for row in cases}),
                 'latest_evaluation_samples': 32 if pilot else 4, 'new_training_for_gallery': bool(pilot),
                 'new_dataset_images_opened': True, 'new_physical_test_images_opened': 0, 'new_cases': expanded['new_cases'] + new_count, 'historical_model_pixels_changed': False,
                 'selection_note': f'{new_count} verified independent-place pilot cases precede {len(expanded["examples"])} metadata-selected old-checkpoint cases and {len(original["examples"])} preserved historical label-selected examples. Each shows the first actual world; none is best-of-K.',
                 'examples_are_old_cohort_diagnostics': not bool(pilot), 'assets': copied,
                 'pilot_publication_state': pilot_state, 'pilot_cases': new_count, 'historical_cases': old_count,
                 'results_cohort': 'new_pilot' if pilot else 'historical', 'result_methods': result_methods,
                 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_paths}}
    data_path = SITE/'data/model_home_zh.json'
    data_path.write_text(json.dumps(home_data, ensure_ascii=False, indent=2)+'\n')
    pilot_note, training_heading, training_description, resource_note = [html.escape(value) for value in training_copy(pilot_status, bool(pilot), candidate_status)]
    resource_paragraph = f'<p class="caption" id="training-resource-policy">{resource_note}</p>' if resource_note else ''
    count = len(cases)
    gallery_intro = (f'新增{new_count}组本轮独立地点验证案例，排在原{old_count}组之前；共{count}组真实输出可切换查看。' if pilot else
                     f'共{count}组真实输出：{sum(row["domain"] == "indoor" for row in cases)}组室内、{sum(row["domain"] == "outdoor" for row in cases)}组室外；均来自此前的检查点。')
    selection_note = f'旧版{len(expanded["examples"])}组按来源、编号和时间固定选择，未按模型效果筛选；最后{len(original["examples"])}组保留此前按标签选定的解释性案例。'
    if pilot:
        selection_note = f'前{new_count}组来自本轮新检查点，五个来源各按编号哈希固定选两组，查询只按输入排序；均显示第1次实际采样。' + selection_note
    scope_note = '旧室内案例排除了与旧训练共享地点的观测，但仍曾参与旧模型选优；旧室外案例使用单独训练的模型，并依赖数据集提供的有效区域。'
    if pilot:
        scope_note = '本轮新案例的父级地点未参与训练或检查点选优；仍是小规模开发验证，非最终测试。' + scope_note + '新旧版本逐例标明，下表只报告本轮统一评估。'
    else:
        scope_note += '图片和旧表格用于诊断，尚不能证明最终测试泛化或超过其它论文。'
    case = cases[0]
    cards = ''.join(image_card(key, label, case['panels'][source], caption) for key, label, source, caption in [
        ('input', '① 模型输入', 'observed', '只有已经观测到的部分；浅灰区域未知。'),
        ('prediction', '② 模型预测', 'correlated_sample', '一次实际采样的完整地图。'),
        ('reference', '③ 真实参考', 'reference', '完整地图，用于核对；不输入推理模型。')])
    thumbnails = ''.join(f'<button data-case="{i}" aria-pressed="{str(i == 0).lower()}" class="case-thumb{" selected" if i == 0 else ""}"><img src="{row["panels"]["correlated_sample"]}" alt="ConPath补全预览 {i+1}：{row["source"]}" loading="lazy"><span>{row["source"]} · {i+1:02d}</span></button>' for i, row in enumerate(cases))
    sources = ''.join(f'<option value="{name}">{name}（{sum(c["source"] == name for c in cases)}组）</option>' for name in sorted({c['source'] for c in cases}))
    labels = PILOT_METHODS if pilot else {'correlated':'ConPath', 'independent':'独立单元模型', 'tiny_deterministic':'确定性补全网络', 'all_floor':'未知全部可通行'}
    rows = []
    for key, name in labels.items():
        if pilot:
            budget = '32' if key in ('correlated', 'independent') else '1'
            m = pilot['result']['methods'][key]['budgets'][budget]
            samples = '—' if key == 'train_radius_prior' else budget
        else:
            m = stats['methods'][key]
            samples = m['samples']
        rows.append(f'<tr data-home-method="{key}"><th scope="row">{name}</th><td>{samples}</td><td>{m["brier"]["mean"]:.4f}</td><td>{m["risk30"]["mean"]*100:.2f}%</td></tr>')
    if pilot:
        result_eyebrow = '02 / 本轮独立地点开发评估'
        result_heading = '地点隔离后，重新比较。'
        result_description = f'本轮{pilot["result"]["validation_parents"]}个独立地点、{pilot["result"]["validation_events"]}个尺寸条件事件；同一查询、同一有效区域，按父级地点平均。随机方法输出32张地图，确定性方法输出1张；训练半径先验没有地图输出。这里只报告本轮结果，旧队列统计保留在研究记录。'
        result_caption = '训练模型为3个随机种子的均值；规则为单次结果。小规模开发验证，非最终测试。两项指标均越低越好。'
        result_reading = ('本轮通过了预设开发筛查；仍需更充分的训练、强外部基线和未参与开发的最终测试。' if pilot['result']['research_screen_passed'] else
                          '本轮未通过全部预设优势筛查；保留负结果，后续改进依据具体差距。小规模开发结果不能证明全面领先。')
        result_link = '<a href="pilot.html#scores">完整7种方法、训练波动与本轮报告 ↗</a>'
    else:
        result_eyebrow = '02 / 既有整体评估'
        result_heading = '有改善，也有明显的不足。'
        result_description = '下表保留旧队列结果；本轮基线通过核验并完整发布后，再切换为本轮结果。下表使用同一批旧查询，随机方法输出4张地图，确定性方法输出1张；与上方32次及128次采样的案例分开报告。'
        result_caption = '三次训练的均值；无训练规则仅一次结果。旧队列开发评估，非最终测试。两项指标均越低越好。'
        result_reading = 'ConPath的概率误差低于独立模型，但“全部可通行”的简单规则更低；该规则的误判风险又更高。当前没有全面领先的证据。'
        result_link = '<a href="research.html#baseline-review">完整6种方法、波动区间与图表 ↗</a>'
    css = sha(SITE/'home.css')[:12]; js = sha(SITE/'home.js')[:12]
    embedded_data = json.dumps(home_data, ensure_ascii=False).replace('<', '\\u003c')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f8faf9"><title>ConPath · 模型效果与实验结果</title><meta name="description" content="直接查看ConPath真实模型输出：输入、预测与参考地图并排比较，保留失败例子。"><link rel="icon" href="favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="home.css?v={css}"><script src="home.js?v={js}" defer></script></head>
<body><a class="skip" href="#effects">跳到模型效果</a><header class="header"><nav class="wrap navigation" aria-label="主导航"><a class="brand" href="index.html">ConPath<span>研究项目</span></a><div><a href="#effects">模型效果</a><a href="#results">实验结果</a><a href="#next">下一步</a><a href="research.html">研究记录 ↗</a></div></nav></header>
<main class="wrap"><section class="intro"><p class="eyebrow">部分地图 → 完整世界 → 通路概率</p><h1>模型补全得怎么样？<br><span>把预测和真实地图放在一起看。</span></h1><p>ConPath根据已观测区域推测完整地图，再结合机器人尺寸估计两个点之间是否有路。</p><p class="status">{gallery_intro}</p><p class="status">{pilot_note}</p></section>
<section id="effects" class="section effects"><span id="method" class="anchor-alias"></span><div class="section-heading"><div><p class="eyebrow">01 / 真实模型输出</p><h2>输入、预测、参考</h2></div><span id="case-badge" class="badge">开发验证 · {count}组案例</span></div>
<div class="controls"><label class="model-select">数据来源<select id="home-source"><option value="all">全部（{count}组）</option>{sources}</select></label><div class="case-paging"><button id="case-prev" aria-label="上一个案例">←</button><span id="case-counter">1 / {count}</span><button id="case-next" aria-label="下一个案例">→</button></div><label class="model-select">模型<select id="home-model"><option value="correlated">ConPath</option><option value="independent">独立单元对照</option></select></label></div>
<div class="case-strip" aria-label="选择ConPath效果案例">{thumbnails}</div><div class="view-row"><div class="segmented light" aria-label="模型输出显示方式"><button data-view="sample" class="selected" aria-pressed="true">一次实际补全</button><button data-view="probability" aria-pressed="false">每格概率图</button></div><p id="view-explanation">绿色表示可通行，深灰表示阻挡。</p></div>
<div class="mobile-panels" aria-label="切换对照图片"><button data-jump="0" class="selected" aria-pressed="true">① 输入</button><button data-jump="1" aria-pressed="false">② 预测</button><button data-jump="2" aria-pressed="false">③ 参考</button></div>
<div id="output-gallery" class="output-gallery">{cards}</div>
<div class="legend" aria-label="地图图例"><span><i class="free"></i>可通行</span><span><i class="blocked"></i>阻挡</span><span><i class="unknown"></i>未知</span><span><i class="outside"></i>有效范围外</span><span><b class="start">● S</b> 起点</span><span><b class="goal">◆ G</b> 目标</span></div>
<div id="case-reading" class="case-reading" aria-live="polite"><div><span>模型估计有路概率</span><strong id="case-probability">{case["event_probability"]["correlated"]*100:.1f}%</strong></div><p id="case-explanation">固定显示第一次实际补全；请对照右侧真实参考查看差异。</p></div>
<p id="case-source" class="caption">{case["source"]} · {case["global_id"]} · {case["split_zh"]} · 机器人半径{case["radius_cells"]}格。{case["checkpoint_scope_zh"]}。</p>
<p class="caption">S、G只是查询端点，图中没有绘制规划路线。{selection_note}所有案例都来自开发阶段，不能代替独立最终测试统计。<a href="research.html#method">查看完整来源与原示例 ↗</a></p>
<aside class="evidence-note"><strong>结果适用范围</strong><p>{scope_note}<a href="research.html#baseline-review">查看旧队列审计与更正 ↗</a></p></aside>
</section>
<section id="results" class="section" data-results-cohort="{home_data['results_cohort']}"><div class="section-heading"><div><p class="eyebrow">{result_eyebrow}</p><h2>{result_heading}</h2></div></div><p class="section-description">{result_description}</p>
<div class="table-scroll"><table><caption>{result_caption}</caption><thead><tr><th scope="col">方法</th><th scope="col">输出数</th><th scope="col">概率误差 ↓</th><th scope="col">30%覆盖率误判 ↓</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="result-reading">{result_reading}</p><p class="caption">概率误差指Brier；30%覆盖率误判指只接受评分最高的30%查询后，实际无路的比例。{result_link}</p></section>
<section id="next" class="section"><div class="section-heading"><div><p class="eyebrow">03 / 下一步</p><h2>先确认优势在干净数据上是否成立。</h2></div></div><ol class="next-steps"><li><span>01</span><div><h3>小规模数据检查已通过</h3><p>100个训练、25个选优、40个验证地点，地点隔离与地图重复检查通过。查询保留目标落在障碍上的自然负例。</p></div></li><li><span>02</span><div><h3>{training_heading}</h3><p>{training_description}</p></div></li><li><span>03</span><div><h3>再验证室外历史融合</h3><p>先修正观测置信度与地面表达，再测试历史几何融合，最后加入注意力；分别检验数据和架构的贡献。</p></div></li></ol>{resource_paragraph}<p class="caption">{pilot_note} <a href="https://github.com/s-team-git/ConPath/blob/main/PARENT_GROUP_PILOT_ZH.md">本轮固定协议 ↗</a>。外部对比优先复用可比较的论文成绩、作者预测或权重。<a href="https://github.com/s-team-git/ConPath/blob/main/WORK_PLAN.md">完整工作计划 ↗</a></p></section>
<section class="records-links" aria-label="深入阅读"><a href="research.html#datasets"><strong>数据集与实景</strong><span>训练数据示例、相机与俯视地图 ↗</span></a><a href="research.html#training-ablations"><strong>内部消融</strong><span>完整模型与两个删减版本 ↗</span></a><a href="research.html#flatlands-external-progress"><strong>外部方法检查</strong><span>短训练输出，尚不是正式对比 ↗</span></a></section>
</main><footer class="wrap footer"><span>ConPath · 可复现研究记录</span><div><a href="research.html">全部研究记录</a><a href="https://github.com/s-team-git/ConPath/blob/main/BASELINE_REVIEW_ZH.md">中文分析</a><a href="https://github.com/s-team-git/ConPath">GitHub</a></div><p>FlatLands派生数据保留上游来源条款。<a href="https://github.com/s-team-git/ConPath/blob/main/DATASET_CHOICE_ZH.md">数据来源与使用说明</a></p></footer>
<dialog id="model-dialog" aria-labelledby="model-dialog-caption"><div class="dialog-toolbar"><p id="model-dialog-caption"></p><button id="model-dialog-close" aria-label="关闭放大图片">关闭 ×</button></div><img id="model-dialog-image" alt="当前模型图片的放大视图"><a id="model-dialog-source" href="#effects">打开原始图片 ↗</a></dialog><p id="home-error" hidden>交互加载失败，仍可查看当前静态示例；请刷新重试。</p>
<script type="application/json" id="home-data">{embedded_data}</script></body></html>'''
    (SITE/'index.html').write_text(page+'\n')
    print(f'Homepage: {count} real model cases, 3 comparison panels, {len(rows)} summary rows; pilot: {pilot_state}.')


if __name__ == '__main__':
    main()
