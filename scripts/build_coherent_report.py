#!/usr/bin/env python3
"""Publish reviewed two-seed comparisons and exportable figures from saved reports."""
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/coherent_parent_pilot_v1'
BASE = ROOT / 'results/parent_group_pilot_v1'
SITE = ROOT / 'site'
ASSETS = SITE / 'assets/zh/coherent-pilot'
SEEDS = (20260910, 20260911)
NAMES = {'coherent_categorical': 'ConPath改进版', 'correlated': '原ConPath',
         'independent': '独立单元对照', 'deterministic': '确定性补全网络'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name):
    for suffix in ('svg', 'pdf'):
        path = ASSETS / f'{name}.{suffix}'
        metadata = {'Date': None} if suffix == 'svg' else {'CreationDate': None, 'ModDate': None}
        fig.savefig(path, bbox_inches='tight', metadata=metadata)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    analysis = json.loads((RUN / 'analysis.json').read_text())
    verified = json.loads((RUN / 'independent_verification.json').read_text())
    assert verified['passed'] is True and verified['analysis_sha256'] == sha(RUN / 'analysis.json')
    assert analysis['training_seeds'] == list(SEEDS) and analysis['final_test'] is False
    assert analysis['validation_reused_for_model_development'] is True
    assert set(analysis['methods']) == set(NAMES)
    status = json.loads((RUN / 'status.json').read_text())
    assert status['stage'] == 'complete'
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.edgecolor': '#dfe7e3', 'text.color': '#203532', 'axes.labelcolor': '#647572',
                         'xtick.color': '#647572', 'ytick.color': '#203532',
                         'svg.fonttype': 'path', 'pdf.fonttype': 42, 'svg.hashsalt': 'conpath-coherent-v1'})
    methods = list(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), sharey=True)
    for ax, metric, xlabel, scale, color in [(axes[0], 'brier', '通路概率误差（Brier）↓', 1, '#137f72'),
                                           (axes[1], 'risk30', '30%覆盖率下的误判风险（%）↓', 100, '#a35e2a')]:
        upper = 0.
        for i, method in enumerate(methods):
            value = analysis['methods'][method][metric]
            ax.errorbar(value['mean']*scale, i, xerr=value['sd']*scale, fmt='o', capsize=4,
                        color=color if method == 'coherent_categorical' else '#899c96', markersize=6, linewidth=1.5)
            label = f'{value["mean"]:.4f}' if scale == 1 else f'{value["mean"]*scale:.2f}%'
            ax.annotate(label, (value['mean']*scale, i), xytext=(6, -16), textcoords='offset points', fontsize=9)
            upper = max(upper, (value['mean'] + value['sd'])*scale)
        ax.set_xlabel(xlabel, labelpad=16); ax.set_xlim(0, upper*1.25)
        ax.grid(axis='x', color='#e9eeeb'); ax.tick_params(axis='y', length=0)
        ax.set_ylim(3.6, -.55)
    axes[0].set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    fig.suptitle('同两个训练种子 · 40个开发地点 · 1,545个通路事件', fontsize=13)
    fig.text(.5, .025, '圆点：两次训练的均值；横向误差棒：样本标准差，不能当作95%置信区间。', ha='center', fontsize=9, color='#647572')
    fig.tight_layout(rect=[0, .07, 1, .97]); save_figure(fig, 'matched-comparison')

    budgets, sources = {}, {}
    for method in methods[:3]:
        reports = []
        for seed in SEEDS:
            path = RUN / 'evaluation' / str(seed) / 'report.json' if method == 'coherent_categorical' else BASE / 'evaluation' / method / str(seed) / 'report.json'
            assert sha(path) == verified['input_file_sha256'][str(path.relative_to(ROOT))]
            reports.append(json.loads(path.read_text()))
            sources[str(path.relative_to(ROOT))] = sha(path)
        budgets[method] = {}
        for k in (4, 32):
            rows = [r['budgets'][str(k)] for r in reports]
            budgets[method][str(k)] = {}
            for metric in ('brier', 'risk30'):
                v = [r[metric] for r in rows]
                budgets[method][str(k)][metric] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'values': v}
            if k == 32:
                for metric in ('brier', 'risk30'):
                    assert abs(budgets[method]['32'][metric]['mean'] - analysis['methods'][method][metric]['mean']) < 1e-12
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    x = np.arange(3); width = .29
    for k, shift, color, label in [(4, -width/2, '#a6bcb4', '4张世界'), (32, width/2, '#137f72', '32张世界')]:
        values = [budgets[m][str(k)]['brier'] for m in methods[:3]]
        means = [v['mean'] for v in values]; deviations = [v['sd'] for v in values]
        ax.bar(x+shift, means, width, yerr=deviations, capsize=4, color=color, label=label,
               error_kw={'linewidth': 1, 'ecolor': '#647572'})
        for px, value, deviation in zip(x+shift, means, deviations):
            ax.text(px, value+deviation+.003, f'{value:.4f}', ha='center', fontsize=9)
    ax.set_xticks(x, [NAMES[m] for m in methods[:3]])
    ax.set_ylabel('通路概率误差（Brier）↓'); ax.set_ylim(0, max(budgets[m][str(k)]['brier']['mean']+budgets[m][str(k)]['brier']['sd'] for m in methods[:3] for k in (4,32))*1.3)
    ax.set_axisbelow(True); ax.grid(axis='y', color='#e9eeeb'); ax.legend(frameon=False, loc='upper right')
    ax.set_title('相同检查点，不同输出数：4张取自同一批32张的前4张', fontsize=12)
    fig.text(.5, .015, '柱高：两个种子的均值；误差棒：标准差。没有额外训练或挑选最好样本。', ha='center', fontsize=9, color='#647572')
    fig.tight_layout(rect=[0, .06, 1, 1]); save_figure(fig, 'sample-budgets')

    rows = []
    for method, name in NAMES.items():
        value = analysis['methods'][method]
        rows.append(f'<tr data-coherent-method="{method}"><th scope="row">{name}</th><td>{value["samples"]}</td><td>{value["brier"]["mean"]:.5f} ± {value["brier"]["sd"]:.5f}</td><td>{value["risk30"]["mean"]*100:.2f}%</td><td>{value["map"]["mean_iou"]:.5f}</td></tr>')
    budget_rows = []
    for method in methods[:3]:
        for k in (4, 32):
            value = budgets[method][str(k)]
            budget_rows.append(f'<tr><th scope="row">{NAMES[method]}</th><td>{k}</td><td>{value["brier"]["mean"]:.5f} ± {value["brier"]["sd"]:.5f}</td><td>{value["risk30"]["mean"]*100:.2f}%</td></tr>')
    seed_rows = []
    for i, seed in enumerate(SEEDS):
        completion_path = RUN / 'runs' / str(seed) / 'complete.json'
        assert sha(completion_path) == verified['input_file_sha256'][str(completion_path.relative_to(ROOT))]
        completed = json.loads(completion_path.read_text())
        seed_rows.append(f'<tr><th scope="row">{seed}</th><td>{completed["epochs"]}</td><td>{completed["best_epoch"]}</td><td>{analysis["methods"]["coherent_categorical"]["brier"]["values"][i]:.5f}</td><td>{analysis["methods"]["correlated"]["brier"]["values"][i]:.5f}</td></tr>')
    old, new = analysis['methods']['correlated'], analysis['methods']['coherent_categorical']
    reduction = 100*(old['brier']['mean']-new['brier']['mean'])/old['brier']['mean']
    risk_difference = 100*(old['risk30']['mean']-new['risk30']['mean'])
    pair = analysis['paired_brier']['correlated']; low, high = pair['ci95']
    conclusion = '通过了预设开发筛查' if analysis['development_screen_passed'] else '未通过预设开发筛查'
    completed_utc = datetime.fromisoformat(status['timestamp_utc']).strftime('%Y-%m-%d %H:%M UTC')
    css = sha(SITE / 'home.css')[:12]
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ConPath · 两种子改进试验结果</title><link rel="stylesheet" href="home.css?v={css}"><link rel="icon" href="favicon.svg"><style>.chart-scroll{{max-width:100%;overflow-x:auto}}.chart-scroll img{{min-width:740px}}</style></head><body>
<header class="header"><nav class="wrap navigation" aria-label="主导航"><a class="brand" href="index.html">ConPath<span>改进试验</span></a><div><a href="index.html#effects">模型效果</a><a href="#scores">本轮结果</a><a href="pilot.html">原基线</a></div></nav></header>
<main class="wrap"><section class="intro"><p class="eyebrow">空间连续采样 / 相同两个种子 / 开发验证</p><h1>通路概率有改善，<br><span>还需要更充分的验证。</span></h1><p>改进版只改变最后一层随机采样的空间相关性，训练数据、优化器和预算沿用原设定。两个种子都已完成，{conclusion}。</p><p class="status">训练和评估完成于{completed_utc}。本轮训练进程已退出，项目显存已释放。</p></section>
<section id="scores" class="section"><div class="section-heading"><h2>相同条件下的比较</h2><span class="badge">两个种子 · 非最终测试</span></div><p class="section-description">四种方法均使用20260910、20260911两个种子；100个训练、25个选优、40个开发验证地点。随机方法输出32张实际地图，确定性网络输出1张。所有方法使用相同查询、有效范围及精确连通性规则。</p><div class="table-scroll"><table><caption>±为两个训练种子的标准差。Brier与误判越低越好，IoU越高越好。</caption><thead><tr><th scope="col">方法</th><th scope="col">输出数</th><th scope="col">通路Brier ↓</th><th scope="col">30%覆盖误判 ↓</th><th scope="col">可通行类IoU ↑</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="result-reading">相比原ConPath，通路概率误差降低{reduction:.2f}%，固定30%覆盖率下的误判降低{risk_difference:.2f}个百分点。可通行类地图IoU由{old['map']['mean_iou']:.5f}降至{new['map']['mean_iou']:.5f}，因此不能说所有指标都有提升。</p><p class="caption">Brier按父级地点等权；并列分数按比例纳入风险计算。IoU只计算未知有效区域的可通行类别，并平均所有采样。<a href="pilot.html#scores">原三种子、七方法基线表</a>另行保留。</p><figure style="margin-top:24px"><div class="chart-scroll"><a href="assets/zh/coherent-pilot/matched-comparison.svg"><img src="assets/zh/coherent-pilot/matched-comparison.svg" alt="同两个种子的四种方法比较，圆点为平均值，误差棒为标准差" style="width:100%;height:auto;display:block"></a></div><figcaption class="caption">绿色和棕色标出改进版，其余为灰色对照。手机可左右滑动图表，点击查看原图。<a href="assets/zh/coherent-pilot/matched-comparison.pdf">下载比较图PDF</a></figcaption></figure></section>
<section class="section"><h2>为什么还不能认定稳定领先</h2><p class="section-description" style="margin-top:16px">原ConPath减改进版的Brier差值为{pair['other_minus_conpath']:.5f}；按来源分层重采样地点后的95%区间为[{low:.5f}, {high:.5f}]，包含0。也就是说，这40个地点给出了正向趋势，但还不能排除地点抽样波动。这一范围只条件于两个训练种子的平均值，没有覆盖全部训练随机性。</p><p class="caption">这40个地点未用于训练或检查点选优，但原基线在它们上的结果已用于选择改进方向，所以它们是反复开发数据。这里没有正式测试或对其它论文的领先声明。</p></section>
<section id="examples" class="section"><h2>直接看同场景的新旧输出</h2><p class="section-description" style="margin-top:16px">首页前10个案例现在可切换“ConPath改进版”“原ConPath”和“独立单元对照”。使用原先固定的场景、输入、起终点及半径，均显示各自的第1张实际补全；概率由同批32张世界计算。成功和失败全部保留。</p><p style="margin-top:16px"><a href="index.html#effects">打开输入／预测／真实参考对照 ↗</a></p><p class="caption">这些图使用种子20260910；总体表使用两个种子，不能用单个案例代表整体收益。绿色为可通行，深灰为阻挡，浅灰为未知；S与G是查询端点，不表示规划路线。</p></section>
<section id="budgets" class="section"><h2>少量采样时会怎样</h2><p class="section-description" style="margin-top:16px">4张输出固定取自同一批32张的前4张，不另选检查点、不挑选最好地图。4张时Brier仍降低，但误判风险由23.73%略升至23.84%，增加0.11个百分点；少量采样下并非两项指标都改善。这是输出预算比较，不是推理耗时排行榜。</p><div class="table-scroll"><table><thead><tr><th>方法</th><th>输出数</th><th>通路Brier ↓</th><th>30%覆盖误判 ↓</th></tr></thead><tbody>{''.join(budget_rows)}</tbody></table></div><figure><div class="chart-scroll"><a href="assets/zh/coherent-pilot/sample-budgets.svg"><img src="assets/zh/coherent-pilot/sample-budgets.svg" alt="三种随机方法在4张与32张输出时的通路概率误差，含两个种子的标准差" style="width:100%;height:auto;display:block"></a></div><figcaption class="caption">浅灰绿柱为4张，深绿柱为32张。手机可左右滑动图表，点击查看原图。<a href="assets/zh/coherent-pilot/sample-budgets.pdf">下载预算比较PDF</a></figcaption></figure></section>
<section class="section"><h2>下一步需要补足的证据</h2><p class="section-description" style="margin-top:16px">本次两种子小实验已经结束。下一阶段先固定更充分的数据规模、收敛检查和强外部基线的共同评估方案，再安排新的有界实验；最终测试需排除全部历史已查看地点后另行冻结。不会把这次开发筛查通过当成论文实验全部完成。</p><details style="margin-top:20px"><summary>展开种子结果、实现限制与复现材料</summary><div class="table-scroll"><table><thead><tr><th>种子</th><th>训练轮次</th><th>选中轮次</th><th>改进版Brier</th><th>原ConPath Brier</th></tr></thead><tbody>{''.join(seed_rows)}</tbody></table></div><p class="caption">最多24轮、600次更新，训练规模仍小；零查询地点的事件微批聚合限制与原基线相同。本轮只改类别采样，没有检验历史关键帧融合或Transformer。参数量同为120,108，但候选增加卷积、CDF和随机数计算，不声称等算力。尺度仅报告格网，输入依赖数据集有效范围。</p><p class="caption"><a href="data/coherent_parent_pilot_zh.json">完整结果JSON</a> · <a href="data/coherent_parent_pilot_verification.json">独立结果核验</a> · <a href="data/coherent_pilot_gallery_verification.json">独立图片核验</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/COHERENT_PILOT_ZH.md">固定协议</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/COHERENT_RESULTS_ZH.md">中文实验记录</a></p></details></section></main><footer class="wrap footer"><a href="index.html#effects">返回模型效果</a><span>ConPath · 两种子开发验证</span></footer></body></html>'''
    (SITE / 'coherent.html').write_text(page+'\n')
    (SITE / 'data/coherent_parent_pilot_zh.json').write_bytes((RUN / 'analysis.json').read_bytes())
    (SITE / 'data/coherent_parent_pilot_verification.json').write_bytes((RUN / 'independent_verification.json').read_bytes())
    report = {'source_analysis_sha256': sha(RUN / 'analysis.json'), 'source_hashes': sources,
              'generator_sha256': sha(Path(__file__)), 'seeds': list(SEEDS), 'budgets': budgets,
              'assets': [{'path': str(p.relative_to(SITE)), 'sha256': sha(p)} for name in ('matched-comparison', 'sample-budgets') for p in (ASSETS/f'{name}.svg', ASSETS/f'{name}.pdf')],
              'final_test': False, 'validation_reused_for_model_development': True}
    (SITE / 'data/coherent_report_zh.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'report_generated': True, 'methods': len(methods), 'figure_exports': 4, 'brier_reduction_percent': reduction}))


if __name__ == '__main__':
    main()
