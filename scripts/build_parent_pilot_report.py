#!/usr/bin/env python3
"""Chinese pilot report and new checkpoint examples from the saved held-out worlds."""
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.parent_pilot_data import load_pilot, sha, stable_rank
from scripts import build_site_visuals as draw
from scripts.evaluate_flatlands_support_clamped import _accelerated_events

RUN = ROOT / 'results/parent_group_pilot_v1'
SITE = ROOT / 'site'
ASSETS = SITE / 'assets/zh/parent-pilot'
NAMES = {'correlated': 'ConPath', 'independent': '独立单元模型', 'deterministic': '确定性补全网络',
         'all_floor': '未知全部可通行', 'all_blocked': '未知全部阻挡', 'nearest_observed': '最近观测规则', 'train_radius_prior': '训练集半径先验'}


def main():
    result = json.loads((RUN / 'analysis.json').read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False, 'axes.edgecolor': '#dfe7e3',
                         'text.color': '#203532', 'axes.labelcolor': '#647572', 'xtick.color': '#647572',
                         'ytick.color': '#203532', 'svg.fonttype': 'path', 'pdf.fonttype': 42})
    methods = list(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, metric, label, scale, color in [(axes[0], 'brier', '通路概率误差（Brier）↓', 1, '#137f72'),
                                           (axes[1], 'risk30', '30%覆盖率下的误判风险（%）↓', 100, '#a35e2a')]:
        for i, method in enumerate(methods):
            budget = '32' if method in ('correlated', 'independent') else '1'
            value = result['methods'][method]['budgets'][budget][metric]
            ax.errorbar(value['mean']*scale, i, xerr=(value['sd'] or 0)*scale, fmt='o' if value['sd'] is not None else 'D',
                        color=color if method == 'correlated' else '#899c96', capsize=3, markersize=6, linewidth=1.4)
            ax.annotate(f'{value["mean"]*scale:.4f}' if scale == 1 else f'{value["mean"]*scale:.2f}%',
                        (value['mean']*scale, i), xytext=(7, -14), textcoords='offset points', fontsize=8)
        ax.set_xlabel(label, labelpad=15); ax.grid(axis='x', color='#e9eeeb', linewidth=.7)
        ax.tick_params(axis='y', length=0); ax.set_xlim(left=0); ax.margins(x=.3)
    axes[0].set_yticks(range(len(methods)), [NAMES[m] for m in methods]); axes[0].invert_yaxis()
    fig.suptitle('40个独立地点 · 同查询、同有效区域的开发比较', fontsize=14, x=.53)
    fig.text(.5, .02, '圆点与误差棒：3个训练种子的均值 ± 标准差；菱形：无训练规则。误差棒不是95%置信区间。', ha='center', fontsize=9, color='#647572')
    fig.tight_layout(rect=[0, .06, 1, .96])
    for suffix in ('svg', 'pdf'):
        fig.savefig(ASSETS / ('baseline-comparison.' + suffix), bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, method in zip(axes, ('correlated', 'independent', 'deterministic')):
        for seed, color in zip((20260910, 20260911, 20260912), ('#137f72', '#b87535', '#597aa8')):
            history = json.loads((RUN / 'runs' / method / str(seed) / 'history.json').read_text())
            ax.plot([r['epoch'] for r in history], [r['calibration']['event_brier'] for r in history], color=color, linewidth=1.3, label=str(seed))
            best = history[-1]['best_epoch']; row = history[best-1]
            ax.scatter([best], [row['calibration']['event_brier']], s=30, color=color)
        ax.set_title(NAMES[method], fontsize=12); ax.set_xlabel('已完成训练轮次'); ax.grid(color='#e9eeeb', linewidth=.7)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel('选优集通路Brier ↓')
    fig.suptitle('选优曲线：原始记录、不平滑；实心点为所选轮次', fontsize=12)
    fig.tight_layout()
    for suffix in ('svg', 'pdf'):
        fig.savefig(ASSETS / ('calibration-curves.' + suffix), bbox_inches='tight')
    plt.close(fig)

    for svg_path in ASSETS.glob('*.svg'):
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')

    samples = load_pilot(RUN / 'data', 'validation')
    chosen = []
    for source in sorted({s.row['source_dataset'] for s in samples}):
        chosen.extend(sorted((s for s in samples if s.row['source_dataset'] == source), key=lambda s: stable_rank('pilot-gallery', s.row['global_id']))[:2])
    worlds = {}
    for method in ('correlated', 'independent'):
        with np.load(RUN / 'evaluation' / method / '20260910/worlds.npz', allow_pickle=False) as z:
            worlds[method] = {gid: w for gid, w in zip(z['global_ids'], z['worlds'])}
    cases = []
    for s in chosen:
        gid = s.row['global_id']; base = 'parent-pilot/' + gid
        # Queries already have a fixed input-only hash ordering in the data cache.
        q = 0; start, goal = s.starts[q].tolist(), s.goals[q].tolist()
        paths = {}
        for key, pixels, title in [('observed', draw.categorical(s.observation, s.valid), '机器人已看到的地图'),
                                   ('reference', draw.categorical(s.observation, s.valid, s.target), '数据集真实参考地图')]:
            paths[key] = draw.map_svg(base + '-' + key, pixels, title, subtitle=s.row['source_dataset'] + ' · ' + gid, points=[start, goal])
        probabilities, first = {}, {}
        for method, label in [('correlated', 'ConPath'), ('independent', '独立单元对照')]:
            w = worlds[method][gid]
            paths[method + '_sample'] = draw.map_svg(base + '-' + method + '_sample', draw.categorical(s.observation, s.valid, w[0]),
                                                    label + ' · 第一次实际补全', subtitle='新基线 · 独立地点验证', points=[start, goal])
            paths[method] = draw.map_svg(base + '-' + method, draw.probability(w.mean(0), s.valid), label + ' · 32次补全的逐格概率',
                                         subtitle='新基线 · 独立地点验证', points=[start, goal], ramp=True)
            events = _accelerated_events(w, s.starts[q:q+1], s.goals[q:q+1], (10,))[:, 0, 0]
            probabilities[method], first[method] = float(events.mean()), bool(events[0])
        cases.append({'id': 'pilot-' + gid, 'global_id': gid, 'source': s.row['source_dataset'], 'scene': s.row['scene_id'],
                      'parent_group': s.row['parent_group'], 'domain': 'indoor', 'cohort': 'new_pilot', 'panels': paths,
                      'split_zh': '新基线 · 独立地点验证', 'checkpoint_scope_zh': '本轮从头训练；此地点未参与训练或检查点选优，属于开发验证',
                      'radius_cells': 10, 'seed': 20260910, 'sampling_seed': 20260910 + 4000000 + int(gid.split('_')[1]),
                      'candidate_index': int(s.candidate_indices[q]), 'start': start, 'goal': goal, 'samples': 32,
                      'target': bool(s.targets[q, 1]), 'event_probability': probabilities, 'displayed_world_event': first,
                      'final_test': False, 'selection': 'two hash-ranked observations/source, first input-hash-ordered query; first sampled world',
                      'checkpoint_selected_on_this_cohort': False})
    gallery = {'examples': cases, 'assets': [{'path': p, 'sha256': sha(SITE / p)} for case in cases for p in case['panels'].values()],
               'checkpoint_seed': 20260910, 'source_analysis_sha256': sha(RUN / 'analysis.json'),
               'saved_worlds': {m: {'path': str((RUN / 'evaluation' / m / '20260910/worlds.npz').relative_to(ROOT)),
                                   'sha256': sha(RUN / 'evaluation' / m / '20260910/worlds.npz')} for m in worlds}}
    (SITE / 'data/parent_pilot_gallery_zh.json').write_text(json.dumps(gallery, ensure_ascii=False, indent=2)+'\n')
    rows = []
    for method, name in NAMES.items():
        budget = '32' if method in ('correlated', 'independent') else '1'
        m = result['methods'][method]['budgets'][budget]
        brier, risk, maps = m['brier'], m['risk30'], m['map']
        sd = f' ± {brier["sd"]:.4f}' if brier['sd'] is not None else ''
        overlap = f'{maps["mean_iou"]:.4f}' if maps is not None else '—'
        rows.append(f'<tr><th scope="row">{name}</th><td>{budget if method != "train_radius_prior" else "—"}</td><td>{brier["mean"]:.4f}{sd}</td><td>{risk["mean"]*100:.2f}%</td><td>{overlap}</td></tr>')
    gate = '通过了预设开发筛查，接下来验证单项模型改进。' if result['research_screen_passed'] else '未通过预设的全部优势筛查；保留负结果，下一项依据具体差距改进。'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ConPath · 独立地点小规模验证</title><link rel="stylesheet" href="home.css?v={sha(SITE/'home.css')[:12]}"><link rel="icon" href="favicon.svg"></head><body><header class="header"><nav class="wrap navigation"><a class="brand" href="index.html">ConPath<span>本轮基线验证</span></a><div><a href="index.html#effects">模型效果</a><a href="#scores">本轮结果</a><a href="research.html">研究记录</a></div></nav></header>
<main class="wrap"><section class="intro"><p class="eyebrow">100个训练地点 / 25个选优地点 / 40个验证地点</p><h1>把地点隔离，再检查模型效果。</h1><p>三种模型各完成三个随机种子的从头训练；最多24轮、600次参数更新。验证包括515个自然查询、1545个尺寸条件事件，其中186个目标点为障碍。</p><p>{gate}</p></section>
<section id="scores" class="section"><div class="section-heading"><h2>同条件比较结果</h2><span class="badge">小规模开发实验</span></div><p class="section-description">通路概率由实际二值世界的精确连通性投票得到；确定性方法只输出1张图。按父级地点平均，全部模型用同一查询及有效区域。下表不与旧队列或论文公开分数混排行。</p><div class="table-scroll"><table><thead><tr><th>方法</th><th>输出数</th><th>通路Brier ↓</th><th>30%覆盖误判 ↓</th><th>可通行类IoU ↑</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="caption">±为3个训练种子的标准差；IoU在未知有效区域内计算可通行类别，并对所有采样取平均。半径先验只拟合训练事件，没有地图输出。</p><figure><a href="assets/zh/parent-pilot/baseline-comparison.svg"><img src="assets/zh/parent-pilot/baseline-comparison.svg" alt="七种方法的通路误差及固定覆盖率风险对照，含训练波动" style="display:block;width:100%;height:auto"></a><figcaption class="caption">绿色、棕色分别标出ConPath的误差、风险；灰色为对照。<a href="assets/zh/parent-pilot/baseline-comparison.pdf">下载PDF</a></figcaption></figure></section>
<section class="section"><h2>本轮新检查点的真实效果</h2><p class="section-description">另新增10组独立地点案例，五个数据来源各2组；按编号固定选择，显示第1次采样，保留失败。与原模型诊断图片分开标明版本。</p><p style="margin-top:16px"><a href="index.html#effects">打开可切换的输入／预测／真实参考图集 ↗</a></p></section>
<section class="section"><h2>这些结果能说明什么</h2><p class="section-description">本轮修正了训练与验证共享父级地点的问题，并保留真实目标为障碍的自然负例。它仍是小预算开发验证：样本只有40个验证地点，不能保证训练已充分收敛，不能替代强外部基线或未参与开发的最终测试。后续改进使用本轮反馈时，这批验证地点也属于反复开发的数据。</p><p class="caption">物理尺度尚未独立核实，只报告格网尺寸；输入使用数据集提供的有效范围。</p><details style="margin-top:24px"><summary>展开训练选优曲线与复现材料</summary><img src="assets/zh/parent-pilot/calibration-curves.svg" alt="三种模型各三个随机种子的完整选优曲线" style="width:100%;height:auto;margin-top:16px"><p class="caption"><a href="data/parent_group_pilot_zh.json">完整数据与按地点配对区间</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/PARENT_GROUP_PILOT_ZH.md">中文固定协议</a> · <a href="assets/zh/parent-pilot/calibration-curves.pdf">训练曲线PDF</a></p></details></section></main><footer class="wrap footer"><a href="index.html">返回模型效果首页</a><span>ConPath · 小规模开发验证，非最终测试</span></footer></body></html>'''
    (SITE / 'pilot.html').write_text(page+'\n')
    print(json.dumps({'pilot_report': True, 'new_checkpoint_cases': len(cases)}))


if __name__ == '__main__':
    main()
