#!/usr/bin/env python3
"""One CPU aggregate-only supplement; copies historic SVG/PDF without alteration."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/paper_figures'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    source = ROOT / 'results/paper_convergence_v1/snapshot_reviewed_v1/paper_validation_snapshot.json'
    snapshot = json.loads(source.read_text())
    assert snapshot['all_current_prediction_numeric_checks_passed']
    assert snapshot['parent_overlap_zero'] and snapshot['physical_archive_split'] == 'train'
    assert snapshot['parent_manifest_counts'] == {'train': 100, 'calibration': 25, 'validation': 40}
    methods = [('correlated', 32, 'ConPath 原版 · K=32', '#3972ab'),
               ('independent', 32, '独立补全 · K=32', '#cf813c'),
               ('deterministic', 1, '确定性补全 · K=1', '#68737f'),
               ('all_floor', 1, '未知全可通行 · K=1', '#809089'),
               ('all_blocked', 1, '未知全障碍 · K=1', '#809089'),
               ('nearest_observed', 1, '最近已观测类别 · K=1', '#809089'),
               ('train_radius_prior', 1, '训练半径先验 · 无地图', '#8c769c')]
    rows = []
    for method, k, _, _ in methods:
        matched = [r for r in snapshot['evidence_rows'] if r['cohort'] == 'parent_isolated_three_seed'
                   and r['method'] == method and r['K'] == k]
        assert len(matched) == 1
        row = matched[0]
        assert row['parents_per_run'] == 40 and row['events_per_run'] == 1545
        expected_seeds = [20260910, 20260911, 20260912] if method in ('correlated', 'independent', 'deterministic') else ['rule']
        assert row['seeds'] == expected_seeds
        rows.append(row)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = BASE / 'rule_controls' / stamp
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.8))
    metrics = [('brier', '事件 Brier ↓'), ('nll', '事件 NLL ↓'), ('ece', '事件 ECE ↓'),
               ('false_safe_at_0_8', '分数 ≥ 0.8 时的误判率 ↓'),
               ('coverage_at_0_8', '分数 ≥ 0.8 时的覆盖率'), ('risk_at_30_percent', '固定 30% 覆盖率的风险 ↓')]
    for ax, (metric, title) in zip(axes.flat, metrics):
        points = []
        for index, (row, (_, _, label, color)) in enumerate(zip(rows, methods)):
            stat = row['metrics'][metric]
            value = stat['mean']
            if value is not None:
                error = stat['sample_sd'] if row['repeats'] > 1 else None
                ax.errorbar(value, index, xerr=error, fmt='o', color=color, capsize=3, markersize=7)
                points.append((index, value, error or 0))
                ax.annotate(f'{value:.4f}', (value, index), xytext=(7, -4), textcoords='offset points', fontsize=9)
            else:
                assert metric == 'false_safe_at_0_8' and row['metrics']['coverage_at_0_8']['mean'] == 0
        upper = max(v + e for _, v, e in points)
        ax.set_xlim(0, max(.1, upper * 1.35))
        for index, row in enumerate(rows):
            if row['metrics'][metric]['mean'] is None:
                ax.text(.02, index, 'N/A：覆盖率为 0', transform=ax.get_yaxis_transform(), va='center', color='#665c70', fontsize=9)
        ax.set_yticks(range(len(rows)), [m[2] for m in methods])
        # Explicit limits retain rows with undefined risk and avoid clipping zero-coverage controls.
        ax.set_ylim(len(rows) - .45, -.55)
        ax.set_title(title, pad=14)
        ax.grid(axis='x', alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('规则控制补图：所有已保存规则均展示；原版 ConPath 没有在所有指标上领先', fontsize=17, y=.985)
    fig.tight_layout(rect=(0, .17, 1, .955), w_pad=3, h_pad=2.4)
    fig.text(.5, .125,
             '地图连续分数另需报告：确定性补全 hidden-map Brier = 0.138685，低于 ConPath 的 0.151468；均为三种子均值。\n'
             '训练半径先验 NLL / ECE 更低，但分数 ≥ 0.8 的覆盖率为 0；风险 N/A 不填 0。不同覆盖率的误判率不能单独排名。\n'
             '训练方法：seed=20260910 / 20260911 / 20260912，均值 ± 种子样本标准差（不是置信区间）；规则：单次确定性结果、无训练种子。',
             ha='center', va='top', fontsize=10, linespacing=1.7)
    fig.text(.5, .045,
             'validation-only / 已复用开发验证：scene=40 个父级地点；query=冻结全部 515 对端点 × radius 0/10/20 grid cell；父级地点等权。\n'
             '0–1 比率（0.13 即 13%）；规则地图仍保留输入证据与支持范围约束。训练半径先验仅输出事件先验，不生成世界。\n'
             f'源 snapshot SHA256={sha(source)[:16]}；协议 SHA256=175d555c6da4dabf；仅已有数值汇总，无新推理。',
             ha='center', va='top', fontsize=9, color='#52616a', linespacing=1.5)
    files = []
    for suffix in ('svg', 'png', 'pdf'):
        p = out / f'09_rule_controls.{suffix}'
        fig.savefig(p, dpi=170, bbox_inches='tight')
        files.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    plt.close(fig)
    write(out / 'values.json', {'selected_rows': rows, 'source': str(source.relative_to(ROOT)), 'sha256': sha(source)})
    shutil.copyfile(__file__, out / 'renderer_source.py')
    caption = ('All available deterministic rule controls and three trained methods on the same 40-parent development cohort. '
               'All rule methods are shown without outcome-based selection. Error bars for trained methods denote the sample SD of three seeds; '
               'rules have one deterministic evaluation. Undefined risk at zero accepted coverage is omitted. '
               'The training-radius prior has lower NLL and ECE but zero coverage at score 0.8. '
               'Deterministic completion has lower continuous hidden-map Brier (0.138685) than original ConPath (0.151468). '
               'No method dominates all objectives; development validation only.')
    manifest = {'schema_version': 1, 'created_utc': stamp, 'kind': 'one_complete_rule_control_supplement',
                'snapshot': {'path': str(source.relative_to(ROOT)), 'sha256': sha(source)},
                'protocol': {'path': 'results/parent_group_pilot_v1/data/protocol.json',
                             'sha256': sha(ROOT / 'results/parent_group_pilot_v1/data/protocol.json')},
                'renderer': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__))},
                'selection': 'all seven existing main-cohort methods, fixed declared order, no outcome selection',
                'files': files, 'english_caption': caption, 'validation_only': True,
                'new_inference': False, 'test_assets_read': False,
                'source_values': {'path': str((out / 'values.json').relative_to(ROOT)), 'sha256': sha(out / 'values.json')}}
    write(out / 'manifest.json', manifest)
    hist = BASE / 'historical_aggregate'
    historical_exists = hist.exists()
    hist.mkdir(exist_ok=True)
    copies = []
    for stem in ('flatlands_clean_marginal_shuffle', 'flatlands_clean_training_ablations'):
        for extension in ('svg', 'pdf'):
            src = ROOT / 'site/assets' / f'{stem}.{extension}'
            dst = hist / src.name
            if historical_exists:
                assert dst.is_file() and sha(src) == sha(dst)
            else:
                shutil.copyfile(src, dst)
            assert sha(src) == sha(dst)
            copies.append({'source': str(src.relative_to(ROOT)), 'destination': str(dst.relative_to(ROOT)),
                           'source_sha256': sha(src), 'destination_sha256': sha(dst), 'byte_identical': True})
    sources = ['site/data/flatlands_clean_paper_analysis.json', 'site/data/flatlands_clean_training_ablations.json',
               'site/data/flatlands_read_scope_erratum.json']
    hm = {'schema_version': 1, 'created_utc': stamp, 'kind': 'historical_aggregate_only_original_byte_copies',
          'files': copies, 'metadata_sources': [{'path': p, 'sha256': sha(ROOT / p)} for p in sources],
          'scope': 'Historical mixed development cohort: 27/160 validation parents overlap training; physical-test archives were historically accessed.',
          'legacy_clean_title_and_test_untouched_flags_superseded_by_erratum': True,
          'current_task_reads_raw_images_or_worlds_or_targets': False,
          'current_task_runs_inference': False, 'new_parent_primary_evidence': False,
          'new_qualitative_selection': False, 'K': 128, 'radii_cells': [0, 10, 20],
          'seeds': [20260831, 20260901, 20260902],
          'rq1': 'Saved exact-marginal shuffle: conditional mechanism evidence in the historical cohort only.',
          'rq2': 'Saved training ablations: no-event retains common event-based checkpoint selection; no-global removes decoder factors but retains encoder context.',
          'no_main_table_or_generalization_authorization': True}
    if not historical_exists:
        write(hist / 'manifest.json', hm)
    else:
        assert json.loads((hist / 'manifest.json').read_text())['files'] == copies
    index = ROOT / 'PAPER_FIGURES.md'
    # Refresh this owned supplement index while preserving old immutable output directories.
    if '\n## 唯一新增规则控制补图\n' in index.read_text():
        index.write_text(index.read_text().split('\n## 唯一新增规则控制补图\n')[0])
    with index.open('a') as handle:
        handle.write('\n## 唯一新增规则控制补图\n\n')
        handle.write(f'[PNG]({out.relative_to(ROOT)}/09_rule_controls.png) · [SVG]({out.relative_to(ROOT)}/09_rule_controls.svg) · [PDF]({out.relative_to(ROOT)}/09_rule_controls.pdf) · [来源与全部数值]({out.relative_to(ROOT)}/manifest.json)\n\n')
        handle.write('原主图03/04只比较训练模型；此图完整纳入已有四个规则控制，不按指标删掉不利结果。训练半径先验的NLL/ECE更低但0.8覆盖率为0；确定性模型的连续隐藏地图Brier为0.138685，低于ConPath的0.151468。风险与覆盖率需一起看。规则没有三次训练，不能制造三种子误差条。\n\n')
        handle.write('English caption: ' + caption + '\n\n')
        handle.write('## 历史RQ1 / RQ2聚合图（单独附录范围）\n\n')
        handle.write('仅原字节复制已有SVG/PDF；不读取旧世界、目标、原始图片，不重算或重新选样。**旧文件名与图标题中的clean不代表地点完全隔离：旧160验证父级地点中27个与训练重叠，且存在历史physical-test归档访问。** 原test-untouched标记已撤回；这些图不能用作新地点泛化主图或外部正式比较。\n\n')
        handle.write('- RQ1 固定边际重排：[SVG](results/paper_figures/historical_aggregate/flatlands_clean_marginal_shuffle.svg) · [PDF](results/paper_figures/historical_aggregate/flatlands_clean_marginal_shuffle.pdf)。English caption: Historical exact-empirical-marginal shuffle diagnostics at K=128. Evidence concerns spatial arrangement conditional on this mixed development cohort; parent overlap and prior physical-test archive access preclude a new-location generalization claim.\n')
        handle.write('- RQ2 训练消融：[SVG](results/paper_figures/historical_aggregate/flatlands_clean_training_ablations.svg) · [PDF](results/paper_figures/historical_aggregate/flatlands_clean_training_ablations.pdf)。English caption: Historical three-seed training ablations on the same mixed development cohort. Removing event loss retains the common event-based checkpoint-selection rule; removing global decoder factors retains encoder global context. These are not current parent-isolated qualitative or final-test results.\n\n')
        handle.write('[历史文件逐字节SHA清单](results/paper_figures/historical_aggregate/manifest.json) · [历史测试访问更正](site/data/flatlands_read_scope_erratum.json)。原按结果挑选的旧定性图仍不采用。\n')
    print(json.dumps({'rule_snapshot': str(out.relative_to(ROOT)), 'manifest_sha256': sha(out / 'manifest.json'),
                      'historical_manifest': str((hist / 'manifest.json').relative_to(ROOT)),
                      'historical_manifest_sha256': sha(hist / 'manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
