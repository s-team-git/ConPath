#!/usr/bin/env python3
"""Publish Chinese ablation evidence only after the six frozen runs finish.

This is a CPU-only report builder. It independently checks scene weighting,
paired Brier intervals, tie handling, source hashes and all seed aggregates.
It neither trains models nor opens a test split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from pathrel.flatlands_eval import join_flatlands_predictions

SEEDS = (20260831, 20260901, 20260902)
LABELS = {'conpath': '完整 ConPath', 'no_event': '去掉事件训练损失',
          'no_global': '去掉解码器全局因子'}
COLORS = {'conpath': '#157f77', 'no_event': '#c78231', 'no_global': '#7862a3'}
DATA = ROOT / 'site/data'
ASSETS = ROOT / 'site/assets/zh'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(actual, expected, context):
    if not np.allclose(actual, expected, atol=1e-12, rtol=0):
        raise ValueError(f'numerical replay mismatch: {context}: {actual} != {expected}')


def replay_metrics(records):
    """Independent equal-scene metrics, including label-independent boundary ties."""
    keys = [r.scene_key for r in records]
    scenes = sorted(set(keys))
    if not scenes:
        raise ValueError('empty event stratum')
    counts = Counter(keys)
    p = np.array([r.probability for r in records], dtype=float)
    y = np.array([r.target for r in records], dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid probability')
    if not set(y).issubset({0., 1.}):
        raise ValueError('nonbinary event target')
    w = np.array([1 / (len(scenes) * counts[key]) for key in keys])
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.minimum(np.searchsorted(np.linspace(0, 1, 11), p, side='right') - 1, 9)
    residual = np.bincount(bins, weights=w * (p - y), minlength=10)
    false_mass = 0.
    remaining = .3
    for probability in np.unique(p)[::-1]:
        tied = p == probability
        mass = float(w[tied].sum())
        fraction = min(1., max(0., remaining / mass))
        false_mass += float(np.sum(w[tied] * (1 - y[tied]))) * fraction
        remaining -= mass * fraction
        if remaining <= 1e-15:
            break
    scene_brier = np.array([
        np.mean([(r.probability - r.target) ** 2 for r in records if r.scene_key == scene])
        for scene in scenes
    ])
    return {'brier': float(np.sum(w * (p - y) ** 2)),
            'nll': float(-np.sum(w * (y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))),
            'ece': float(np.abs(residual).sum()), 'risk30': false_mass / .3,
            'positive_rate': float(np.sum(w * y)), 'mean_probability': float(np.sum(w * p)),
            'scene_brier': scene_brier}


def interval_conclusion(intervals):
    if all(lo > 0 for lo, hi in intervals):
        return '三次训练的区间均高于零，支持完整模型在此验证设置下更好。'
    if all(hi < 0 for lo, hi in intervals):
        return '三次训练的区间均低于零，去掉该部分反而更好，当前结果不支持保留它的收益。'
    return '区间包含零或不同训练之间方向不一致，尚未证实完整模型有稳定优势。'


def verify(root, report):
    progress = read(root / 'progress.json')
    expected = {(variant, seed) for variant in ('no_event', 'no_global') for seed in SEEDS}
    if progress['status'] != 'complete' or len(progress['runs']) != 6:
        raise ValueError('the full matrix and analysis must finish before publication')
    if {(r['variant'], r['seed']) for r in progress['runs'] if r['status'] == 'complete'} != expected:
        raise ValueError('completed runs do not match the frozen six-run matrix')
    if not report['validation_only'] or report['test_evaluated'] is not False or progress['test_evaluated'] is not False:
        raise ValueError('validation-only boundary missing')
    protocol = report['protocol']
    if (protocol['seeds'] != list(SEEDS) or protocol['events_per_seed'] != 4224
            or protocol['contributing_scenes'] != 142 or protocol['bootstrap_samples'] != 2000
            or protocol['bootstrap_seed'] != 20260907):
        raise ValueError('frozen statistical protocol mismatch')
    if len(report['audits']) != 6 or {(r['variant'], r['seed']) for r in report['audits'] if r['passed']} != expected:
        raise ValueError('six successful run audits are required')
    if sha(root / 'matrix.json') != report['matrix_sha256']:
        raise ValueError('matrix hash mismatch')
    if sha(ROOT / 'scripts/analyze_flatlands_clean_ablations.py') != report['script_sha256']:
        raise ValueError('analysis source hash mismatch')
    if set(report['methods']) != set(LABELS):
        raise ValueError('missing or unexpected method')
    canonical = None
    replay = {}
    source_files = {}
    for variant in LABELS:
        method = report['methods'][variant]
        if [r['seed'] for r in method['seeds']] != list(SEEDS):
            raise ValueError('seed order/count mismatch')
        for row in method['seeds']:
            for key in ('prediction', 'run'):
                path = ROOT / row[key]
                if sha(path) != row[key + '_sha256']:
                    raise ValueError(f'{key} hash mismatch: {path}')
                source_files[row[key]] = row[key + '_sha256']
            run = read(ROOT / row['run'])
            if run['test_evaluated'] is not False:
                raise ValueError('unexpected test evaluation')
            config = run['config']
            records, _ = join_flatlands_predictions(ROOT / row['prediction'], ROOT / config['selection'],
                                                    ROOT / config['queries'], split='validation')
            records = tuple(sorted(records, key=lambda r: r.key))
            keys = [(r.key, r.scene_key, r.target) for r in records]
            canonical = keys if canonical is None else canonical
            if keys != canonical or len(records) != 4224 or len({r.scene_key for r in records}) != 142:
                raise ValueError('same-event/scene/label contract mismatch')
            actual = replay_metrics(records)
            replay[variant, row['seed']] = actual
            for metric in ('brier', 'nll', 'ece', 'positive_rate', 'mean_probability'):
                close(actual[metric], row['metrics'][metric], (variant, row['seed'], metric))
            close(actual['risk30'], row['risk_at_30_percent']['false_safe_rate'], (variant, row['seed'], 'risk30'))
            for stratum, metrics in row['strata'].items():
                kind, value = stratum.split('/', 1)
                subset = tuple(r for r in records if (str(r.radius_cells) == value if kind == 'radius' else r.source_dataset == value))
                checked = replay_metrics(subset)
                for metric in ('brier', 'nll', 'ece', 'positive_rate', 'mean_probability'):
                    close(checked[metric], metrics[metric], (variant, row['seed'], stratum, metric))
        for metric in ('brier', 'nll', 'ece', 'risk30'):
            values = [replay[variant, seed][metric] for seed in SEEDS]
            aggregate = method['risk_at_30_percent'] if metric == 'risk30' else method['aggregate'][metric]
            close(aggregate['mean'], np.mean(values), (variant, metric, 'mean'))
            close(aggregate['sample_sd'], np.std(values, ddof=1), (variant, metric, 'sample SD'))
            close(aggregate['values'], values, (variant, metric, 'seed values'))
    # Average scene-level paired errors in every whole-scene draw. Event rows are
    # never resampled as independent observations, and seeds are never pooled.
    draws = np.random.default_rng(20260907).integers(0, 142, (2000, 142))
    for variant in ('no_event', 'no_global'):
        deltas = []
        if [r['seed'] for r in report['paired'][variant]['seeds']] != list(SEEDS):
            raise ValueError('paired seed order/count mismatch')
        for row in report['paired'][variant]['seeds']:
            seed = row['seed']
            delta = replay[variant, seed]['scene_brier'] - replay['conpath', seed]['scene_brier']
            point, interval = float(delta.mean()), np.quantile(delta[draws].mean(axis=1), [.025, .975])
            paired = row['event_metrics']['brier']
            close(paired['independent_minus_correlated'], point, (variant, seed, 'paired direction'))
            close(paired['bootstrap_95'], interval, (variant, seed, 'scene CI'))
            close(row['equal_coverage_risk']['0.3']['comparator_minus_conpath'],
                  replay[variant, seed]['risk30'] - replay['conpath', seed]['risk30'], (variant, seed, 'risk direction'))
            deltas.append(point)
        aggregate = report['paired'][variant]['brier_delta']
        close(aggregate['mean'], np.mean(deltas), (variant, 'paired mean'))
        close(aggregate['sample_sd'], np.std(deltas, ddof=1), (variant, 'paired sample SD'))
    return {'passed': True, 'test_evaluated': False, 'source_files': source_files,
            'prediction_files_replayed': 9, 'same_events_per_run': 4224, 'scenes': 142,
            'independent_brier_interval_replays': 6, 'bootstrap_samples_per_interval': 2000,
            'checks': ['Brier/NLL/ECE from CSV', 'equal-scene weighting', 'source/radius strata',
                       'fractional label-independent ties at 30% coverage', 'three-seed mean and sample SD',
                       'whole-scene paired Brier intervals', 'ablation minus full direction', 'source hashes']}


def fmt(item, percent=False):
    factor = 100 if percent else 1
    precision = 2 if percent else 5
    return f"{item['mean'] * factor:.{precision}f} ± {item['sample_sd'] * factor:.{precision}f}" + ('%' if percent else '')


def verify_cases(root, analysis):
    import base64
    import io
    import xml.etree.ElementTree as ET
    from PIL import Image
    from pathrel.labels import clearance_radius_map
    from scripts.build_site_visuals import categorical, flat_dataset, probability
    from scripts.render_flatlands_k128_advantage import _event

    report = read(root / 'report.json')
    if (not report['passed'] or report['test_evaluated'] is not False or report['training_started'] is not False
            or report['seed'] != 20260831 or len(report['assets']) != 12):
        raise ValueError('completed, validation-only fixed-case rendering required')
    if [(r['global_id'], r['candidate_index'], r['radius_cells']) for r in report['cases']] != [('obs_266762', 23, 10), ('obs_249512', 5, 20)]:
        raise ValueError('fixed visual cases changed')
    for path, digest in report['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'visual source changed: {path}')
    for asset in report['assets']:
        if Path(asset['path']).parent != Path('assets/zh') or sha(root / Path(asset['path']).name) != asset['sha256']:
            raise ValueError('visual asset path/hash mismatch')
    arrays = report['posterior_arrays']
    if sha(ROOT / arrays['path']) != arrays['sha256']:
        raise ValueError('visual posterior array hash mismatch')
    saved = np.load(ROOT / arrays['path'])
    dataset = flat_dataset('validation')
    by_id = {r.global_id: i for i, r in enumerate(dataset.observations)}
    for case in report['cases']:
        sample = dataset[by_id[case['global_id']]]
        for variant in LABELS:
            mean = saved[case['id'] + '_' + variant]
            world = saved[case['id'] + '_' + variant + '_first_world']
            if world.dtype != bool or world.shape != sample.epistemic_mask.shape or np.any(world & ~sample.epistemic_mask):
                raise ValueError('invalid saved world/support')
            center_free = (clearance_radius_map(world) >= case['radius_cells']) & sample.epistemic_mask
            event = _event(world, tuple(case['start']), tuple(case['goal']), case['radius_cells'])
            if event != case['first_world_event'][variant]:
                raise ValueError('displayed first-world connectivity mismatch')
            for paths, expected in ((case['panels'], probability(mean, sample.epistemic_mask)),
                                    (case['footprint_panels'], categorical(sample.input_bev, sample.epistemic_mask, center_free))):
                image = ET.parse(root / Path(paths[variant]).name).find('{http://www.w3.org/2000/svg}image')
                raw = base64.b64decode(image.attrib['href'].split(',', 1)[1])
                pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
                if not np.array_equal(pixels, expected):
                    raise ValueError('published map pixels differ from saved posterior/footprint geometry')
    saved.close()
    dataset.close()
    for variant in LABELS:
        source = analysis['methods'][variant]['seeds'][0]
        config = read(ROOT / source['run'])['config']
        records, _ = join_flatlands_predictions(ROOT / source['prediction'], ROOT / config['selection'],
                                                ROOT / config['queries'], split='validation')
        records = {r.key: r for r in records}
        for case in report['cases']:
            record = records[case['global_id'], case['candidate_index'], case['radius_cells']]
            close(case['event_probability'][variant], record.probability, (variant, case['id'], 'displayed query probability'))
            if case['target'] != record.target:
                raise ValueError('visual case target mismatch')
    return report


def figures(report):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 12, 'axes.unicode_minus': False,
                         'text.color': '#263544', 'axes.labelcolor': '#667582',
                         'xtick.color': '#667582', 'ytick.color': '#263544',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'conpath-training-ablation-zh-v1'})
    assets = []
    def save(fig, stem, mobile):
        for suffix in (('svg',) if mobile else ('svg', 'pdf')):
            path = ASSETS / (stem + ('-mobile' if mobile else '') + '.' + suffix)
            metadata = {'Date': None} if suffix == 'svg' else {'CreationDate': None, 'ModDate': None}
            fig.savefig(path, metadata=metadata, facecolor='white')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            assets.append({'path': str(path.relative_to(ROOT / 'site')), 'sha256': sha(path)})
        plt.close(fig)
    for mobile in (False, True):
        fig, ax = plt.subplots(figsize=(4.3, 5.7) if mobile else (11, 4.7))
        fig.subplots_adjust(left=.10 if mobile else .29, right=.93, bottom=.25 if mobile else .18, top=.72 if mobile else .76)
        stats = [report['methods'][k]['aggregate']['brier'] for k in LABELS]
        means = [s['mean'] for s in stats]; deviations = [s['sample_sd'] for s in stats]
        positions = np.arange(3)
        ax.barh(positions, means, height=.38 if mobile else .43, color=list(COLORS.values()),
                xerr=deviations, error_kw={'ecolor': '#354757', 'capsize': 4, 'elinewidth': 1.2})
        extent = max(m + s for m, s in zip(means, deviations))
        for index, (key, statistic) in enumerate(zip(LABELS, stats)):
            ax.text(statistic['mean'] + statistic['sample_sd'] + extent * .035, index,
                    f"{statistic['mean']:.4f}", va='center', fontsize=12)
            if mobile:
                ax.text(0, index - .31, LABELS[key], fontsize=12, color=COLORS[key])
        ax.set(yticks=[] if mobile else positions, xlim=(0, extent * 1.35),
               ylim=(2.55, -.7), xlabel='路径概率误差 Brier ↓（越低越好）')
        if not mobile:
            ax.set_yticklabels(list(LABELS.values()))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='x', color='#e8edef', linewidth=.8); ax.set_axisbelow(True)
        fig.text(.055, .94, '训练消融：每次只去掉一个部分', fontsize=16 if mobile else 19, weight='bold', va='top')
        fig.text(.055, .85, '青绿：完整模型 · 橙色：无事件损失\n紫色：无全局因子' if mobile else '青绿：完整模型 · 橙色：无事件损失 · 紫色：无全局因子', fontsize=10 if mobile else 12, va='top')
        fig.text(.055, .075, '柱长：3 次训练的平均误差。\n横线：训练种子标准差，不是置信区间。\n4,224 个验证事件 / 142 个场景 / K=128。' if mobile else '柱长：3 次训练的平均误差；横线：训练种子标准差。\n相同 4,224 个验证事件 / 142 个场景；每次采样 128 张地图。', fontsize=10, color='#667582', va='center')
        save(fig, 'training-ablation-brier', mobile)

        fig, ax = plt.subplots(figsize=(4.3, 7.5) if mobile else (11, 5.4))
        fig.subplots_adjust(left=.12 if mobile else .30, right=.94, bottom=.23 if mobile else .20, top=.74 if mobile else .77)
        row_index = 0; labels = []; endpoints = [0.]
        for variant in ('no_event', 'no_global'):
            for row in report['paired'][variant]['seeds']:
                delta = row['event_metrics']['brier']; low, high = delta['bootstrap_95']
                value = delta['independent_minus_correlated']
                ax.plot([low, high], [row_index, row_index], color=COLORS[variant], linewidth=2.2)
                ax.plot([low, high], [row_index, row_index], '|', color=COLORS[variant], markersize=8)
                ax.plot(value, row_index, 'o', color=COLORS[variant], markersize=6)
                label = f"{'无事件损失' if variant == 'no_event' else '无全局因子'} · {row['seed']}"
                labels.append(label)
                if mobile:
                    ax.text(0, row_index - .33, label, transform=ax.get_yaxis_transform(), fontsize=11, color=COLORS[variant])
                endpoints += [low, high]; row_index += 1
        ax.axvline(0, color='#8b98a1', linewidth=1.2, linestyle='--')
        span = max(endpoints) - min(endpoints) or .01
        ax.set(xlim=(min(endpoints) - span * .08, max(endpoints) + span * .08),
               ylim=(5.5, -.75), yticks=[] if mobile else range(6),
               xlabel='误差差值：去掉该部分 − 完整模型')
        if not mobile:
            ax.set_yticklabels(labels, fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='x', color='#edf0f2', linewidth=.8); ax.set_axisbelow(True)
        fig.text(.055, .94, '差异在不同场景中稳定吗？', fontsize=16 if mobile else 19, weight='bold', va='top')
        fig.text(.055, .85, '橙色：无事件损失 · 紫色：无全局因子\n每行一次独立训练，数字为种子编号。' if mobile else '橙色：去掉事件损失 · 紫色：去掉解码器全局因子\n每行对应一次独立训练，数字是训练种子编号。', fontsize=10 if mobile else 12, va='top')
        fig.text(.055, .075, '圆点：差值；横线：95% 场景重采样区间。\n整段在零右侧：完整模型更好。\n跨零：差异尚不明确；每个种子重采样 2,000 次。\n当前为验证集描述性区间。' if mobile else '圆点：误差差值；横线：95% 场景重采样区间。\n整段在零右侧：完整模型更好；跨零：差异尚不明确。\n每个种子重采样 2,000 次；这是验证集的描述性区间。', fontsize=10, color='#667582', va='center')
        save(fig, 'training-ablation-paired', mobile)
    return assets


def markdown(report, root):
    methods = report['methods']
    lines = ['# 本轮评估汇总（2026-09-08）', '',
             '六组训练消融及其完整精确评估已完成，并与三组已冻结的完整 ConPath 逐种子配对。'
             '使用冻结的 160 个训练观测和 160 个验证观测，其中 142 个验证场景贡献了保留的路径事件。'
             '这里只汇报实际验证结果，正式测试集尚未评估。', '',
             '## 1. 先看总表', '',
             '每个模型独立训练三次。数值为均值 ± **训练种子标准差**；不是置信区间。所有指标越低越好。', '',
             '| 模型 | 路径概率误差 Brier ↓ | NLL ↓ | 校准误差 ECE ↓ | 接受 30% 查询后的无路误判比例 ↓ |',
             '|---|---:|---:|---:|---:|']
    for variant, method in methods.items():
        lines.append('| ' + LABELS[variant] + ' | ' + ' | '.join(
            [fmt(method['aggregate'][metric]) for metric in ('brier', 'nll', 'ece')]
            + [fmt(method['risk_at_30_percent'], True)]) + ' |')
    lines += ['', '**指标含义：** Brier 是概率与真实有路/无路标签的平方偏差；NLL 对错误且过度自信的判断惩罚更重；'
              'ECE 衡量十个概率分箱内的信心与实际频率差异。30% 风险在相同接受比例下比较；边界同分查询按不使用标签的相同比例接受。', '',
              '![三种模型的中文消融结果图](site/assets/zh/training-ablation-brier.svg)', '', '## 2. 两个部分分别有什么作用', '']
    for variant in ('no_event', 'no_global'):
        comparison = report['paired'][variant]
        intervals = [r['event_metrics']['brier']['bootstrap_95'] for r in comparison['seeds']]
        lines += [f"**{LABELS[variant]}：** 平均 Brier 差值为 {fmt(comparison['brier_delta'])}（消融 − 完整）。"
                  + interval_conclusion(intervals), '']
    lines += ['“去掉事件训练损失”只把事件损失权重设为零，仍使用同样的验证事件选最佳检查点；'
              '“去掉全局因子”只关闭解码器的全局随机因子，保留编码器全局上下文和局部相关噪声。'
              '因此，后者的结果不能解释成“所有空间相关性都有用/没用”。', '',
              '| 消融 | 训练种子 | Brier 差值（消融 − 完整） | 95% 场景配对区间 | 30% 风险差值（百分点） | 风险差值 95% 区间（百分点） |',
              '|---|---:|---:|---:|---:|---:|']
    for variant in ('no_event', 'no_global'):
        for row in report['paired'][variant]['seeds']:
            event = row['event_metrics']['brier']; risk = row['equal_coverage_risk']['0.3']
            lo, hi = event['bootstrap_95']; rlo, rhi = risk['bootstrap_95']
            lines.append(f"| {LABELS[variant]} | {row['seed']} | {event['independent_minus_correlated']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | "
                         f"{risk['comparator_minus_conpath'] * 100:+.2f} | [{rlo * 100:+.2f}, {rhi * 100:+.2f}] |")
    risk_intervals = [row['equal_coverage_risk']['0.3']['bootstrap_95']
                      for comparison in report['paired'].values() for row in comparison['seeds']]
    if all(lo <= 0 <= hi for lo, hi in risk_intervals):
        lines += ['', '**误判风险仍没有稳定改善的证据。** 两项消融的六个 30% 覆盖率风险区间都包含零；'
                  '因此，本轮可以支持总体概率误差改善，不能据此写成“安全性稳定提升”。']
    lines += ['', '![逐种子差值与场景配对区间](site/assets/zh/training-ablation-paired.svg)', '',
              '每个种子内，以整个场景为单位重采样 2,000 次（随机种子 20260907），而非把 4,224 个事件视作独立样本。'
              '三次训练的波动与场景区间分别报告，没有把预测概率平均成一个新集成模型。'
              '本批次沿用已参与检查点选择的验证集，区间仅作描述，不是经过多重比较校正的确认性检验。', '',
              '## 3. 全部训练结果与分层表现', '',
              '| 模型 | 训练种子 | 训练停止轮次 | 选中轮次 | 最终精确 Brier | NLL | ECE | 30% 风险 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    progress = read(root / 'progress.json')
    by_run = {(r['variant'], r['seed']): r for r in progress['runs']}
    audits = {(r['variant'], r['seed']): r for r in report['audits']}
    for variant, method in methods.items():
        for row in method['seeds']:
            run = read(ROOT / row['run'])
            if variant != 'conpath':
                epoch = by_run[variant, row['seed']]['last_epoch']['epoch']
                best = audits[variant, row['seed']]['checkpoint']['epoch']
            else:
                history = run.get('history', [])
                epoch = history[-1]['epoch'] if history else '原已冻结'
                best = run['selection']['best_epoch']
            values = ' | '.join(f"{row['metrics'][m]:.5f}" for m in ('brier', 'nll', 'ece'))
            lines.append(f"| {LABELS[variant]} | {row['seed']} | {epoch} | {best} | {values} | {row['risk_at_30_percent']['false_safe_rate'] * 100:.2f}% |")
    lines += ['', '下表在每个来源或半径内部重新令场景等权，再汇总三次训练；未给每个分层追加显著性主张。'
              '半径先用“格”报告，FlatLands 论文名义分辨率与本地元数据的差异仍待核对。', '',
              '| 分层 | 完整 ConPath Brier | 无事件损失 Brier | 无全局因子 Brier |', '|---|---:|---:|---:|']
    for stratum in methods['conpath']['seeds'][0]['strata']:
        name = stratum.replace('radius/', '机器人半径 / ') + (' 格' if stratum.startswith('radius/') else '')
        name = name.replace('source/', '数据来源 / ')
        values = []
        for variant in LABELS:
            scores = [r['strata'][stratum]['brier'] for r in methods[variant]['seeds']]
            values.append(fmt({'mean': np.mean(scores), 'sample_sd': np.std(scores, ddof=1)}))
        lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    if all(r['strata']['radius/20']['mean_probability'] == 0 for r in methods['no_event']['seeds']):
        prevalence = methods['no_event']['seeds'][0]['strata']['radius/20']['positive_rate']
        lines += ['', f'**大半径的漏判是一个明确失败模式。** 去掉事件训练损失后，半径 20 格的所有验证预测均为 0，'
                  f'但该层有 {prevalence:.2%} 的事件实际有路（场景等权）。这解释了为什么单元格概率图即使很绿，'
                  '考虑机器人尺寸后的完整世界仍可能普遍无路。半径分层属于诊断，不能据此扩展尚未核对的物理尺度主张。']
    lines += ['', '## 4. 看两个固定地图示例', '',
              '以下仍是网站此前选定的两个验证示例，未根据本轮消融结果重新挑图。只展示种子 20260831；'
              '每幅图都来自真实已训练检查点，米白到青绿统一表示每个单元格的可通行概率从 0 到 1。'
              '蓝色圆点 S 是起点，棕色菱形 G 是目标；没有把两个点画成规划路线。', '']
    cases = read(DATA / 'training_ablation_cases_zh.json')['cases']
    for case in cases:
        lines += [f"**{case['global_id']}：参考地图{'有路' if case['target'] else '无路'}，半径 {case['radius_cells']} 格。**", '',
                  f"[查看模型输入](site/{case['observed']}) · [查看完整参考](site/{case['reference']})", '',
                  '| 完整 ConPath | 去掉事件训练损失 | 去掉解码器全局因子 |', '|---|---|---|',
                  '| ' + ' | '.join(f"![{LABELS[v]}](site/{case['panels'][v]})" for v in LABELS) + ' |', '',
                  '该查询在原精确验证中的有路概率：' + '；'.join(f"{LABELS[v]} {case['event_probability'][v]:.1%}" for v in LABELS) + '。', '',
                  f"下面固定展示**第 1 次实际采样，按机器人半径 {case['radius_cells']} 格收缩后的图**。绿色是机器人中心可放置区域，深灰不可放置；这是一个完整世界里的判断，不是 128 次的平均概率。", '',
                  '| 完整 ConPath | 去掉事件训练损失 | 去掉解码器全局因子 |', '|---|---|---|',
                  '| ' + ' | '.join(f"![{LABELS[v]}的首个采样及机器人尺寸](site/{case['footprint_panels'][v]})" for v in LABELS) + ' |', '']
    lines += ['地图颜色由固定的新绘图随机流生成，三个模型均为 K=128、每批 8 张；标题中的通路概率直接来自原精确验证 CSV，二者随机流不同。'
              '单元格概率图与整体通路概率不同，两个解释性示例也不能代替全验证集的统计。', '',
              '## 5. 论文能写什么，下一步补什么', '',
              '本轮补齐了 **FlatLands 有界训练集上的三种子训练消融**，可以据实际方向讨论事件训练目标与全局解码因子的作用。'
              '“打散空间结构”的旧实验仍是固定边缘概率的评估时干预，两者回答的问题不同。', '',
              '已有独立单元对照为 0.09521，同检查点的强均值地图对照为 0.06957；后者与完整模型的配对区间都包含零。'
              '这些强对照及既有 30% 覆盖率风险的无明确优势结果仍须保留，不能只展示有利消融。'
              'UnScenes3D 当前仍是观测误差下界诊断，不能写成跨域成功。', '',
              '接下来按已确认方案推进：先核对 FlatLands 尺度和外部模型输入，完成固定训练样本的成本测量；'
              '然后比较 LaMa/集成、条件流匹配，并在 CogniPlan 原生地图上比较官方生成模块。'
              '新主表统一真实 K=4 输出及输入/通路几何，另报采样量与耗时成本曲线，保留强确定性和直接事件对照。'
              '如果扩大训练集，所有主表方法要在同一冻结版本重新训练。正式测试应在完整协议冻结后统一进行。', '',
              '这些外部主比较尚未运行。本轮内部消融完成不等于整篇论文实验已经完成。', '',
              '## 6. 复现与图片下载', '',
              '- [机器可读完整统计及预测哈希](site/data/flatlands_clean_training_ablations.json)',
              '- [每次训练结果 CSV](site/data/training_ablation_metrics.csv)',
              '- [逐种子配对差值 CSV](site/data/training_ablation_paired.csv)',
              '- [中文主图 PDF](site/assets/zh/training-ablation-brier.pdf) · [中文配对区间 PDF](site/assets/zh/training-ablation-paired.pdf)',
              '- [冻结的消融协议](CLEAN_ABLATION_PLAN.md) · [后续实验方案](EXPERIMENT_DESIGN_ZH.md)', '',
              '```bash', 'PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/build_ablation_summary_zh.py',
              '/home/hairo/miniconda3/bin/python3.13 scripts/build_site_page.py', '```', '',
              '上述命令只重建已经完成并通过检查的验证汇总，不启动训练。原始地图、权重与完整预测留在本地结果目录，Git 保存汇总、代码和图表。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-root', type=Path, default=ROOT / 'results/paper_clean_ablation_matrix_v1')
    parser.add_argument('--audit-output', type=Path, default=ROOT / 'results/ablation_summary_current_audit.json')
    parser.add_argument('--cases-root', type=Path, default=ROOT / 'results/ablation_publication_20260908/cases_v2')
    args = parser.parse_args()
    root = args.matrix_root.resolve()
    if not (root / 'analysis/report.json').is_file():
        parser.error('完整评估报告尚未生成；不会发布部分数据，也不会启动新训练。')
    report = read(root / 'analysis/report.json')
    verification = verify(root, report)
    cases = verify_cases(args.cases_root, report)
    ASSETS.mkdir(parents=True, exist_ok=True)
    assets = figures(report)
    for asset in cases['assets']:
        shutil.copyfile(args.cases_root / Path(asset['path']).name, ROOT / 'site' / asset['path'])
    shutil.copyfile(args.cases_root / 'report.json', DATA / 'training_ablation_cases_zh.json')
    public_report = DATA / 'flatlands_clean_training_ablations.json'
    shutil.copyfile(root / 'analysis/report.json', public_report)
    (ROOT / 'EVALUATION_SUMMARY_ZH.md').write_text(markdown(report, root))
    for suffix in ('svg', 'pdf'):
        shutil.copyfile(root / f'analysis/figures/flatlands_clean_training_ablations.{suffix}',
                        ROOT / f'site/assets/flatlands_clean_training_ablations.{suffix}')
    from scripts.build_paper_evidence import training_ablation_evidence
    evidence_path = ROOT / 'PAPER_EVIDENCE.md'
    evidence = evidence_path.read_text()
    start, end = '<!-- TRAINING_ABLATIONS_START -->', '<!-- TRAINING_ABLATIONS_END -->'
    block = '\n'.join(training_ablation_evidence())
    if start in evidence:
        before, old = evidence.split(start, 1)
        _, after = old.split(end, 1)
        evidence = before + block.rstrip() + after
    else:
        evidence = evidence.replace('## Reproduce', block + '\n## Reproduce')
    evidence_path.write_text(evidence)
    with (DATA / 'training_ablation_metrics.csv').open('w', newline='') as handle:
        out = csv.writer(handle, lineterminator='\n')
        out.writerow(['variant', 'label_zh', 'seed', 'brier', 'nll', 'ece', 'risk_at_30_percent', 'prediction_sha256'])
        for variant, method in report['methods'].items():
            for row in method['seeds']:
                out.writerow([variant, LABELS[variant], row['seed'], *[row['metrics'][k] for k in ('brier', 'nll', 'ece')],
                              row['risk_at_30_percent']['false_safe_rate'], row['prediction_sha256']])
    with (DATA / 'training_ablation_paired.csv').open('w', newline='') as handle:
        out = csv.writer(handle, lineterminator='\n')
        out.writerow(['variant', 'label_zh', 'seed', 'brier_delta_ablation_minus_full', 'brier_ci_low', 'brier_ci_high',
                      'risk30_delta_ablation_minus_full', 'risk30_ci_low', 'risk30_ci_high'])
        for variant, comparison in report['paired'].items():
            for row in comparison['seeds']:
                brier = row['event_metrics']['brier']; risk = row['equal_coverage_risk']['0.3']
                out.writerow([variant, LABELS[variant], row['seed'], brier['independent_minus_correlated'], *brier['bootstrap_95'],
                              risk['comparator_minus_conpath'], *risk['bootstrap_95']])
    manifest = {'schema_version': 1, 'validation_only': True, 'test_evaluated': False,
                'report': str(public_report.relative_to(ROOT / 'site')), 'report_sha256': sha(public_report),
                'builder_sha256': sha(Path(__file__)), 'labels': LABELS, 'colors': COLORS, 'assets': assets,
                'summary': {variant: interval_conclusion([r['event_metrics']['brier']['bootstrap_95'] for r in comparison['seeds']])
                            for variant, comparison in report['paired'].items()}}
    write(DATA / 'training_ablation_visuals_zh.json', manifest)
    verification['report_sha256'] = sha(public_report)
    verification['builder_sha256'] = sha(Path(__file__))
    verification['assets'] = assets
    verification['fixed_case_count'] = len(cases['cases'])
    verification['fixed_case_assets'] = cases['assets']
    verification['fixed_case_raster_replays'] = 12
    verification['first_world_connectivity_replays'] = 6
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    write(args.audit_output, verification)
    print(json.dumps({'passed': True, 'report': 'EVALUATION_SUMMARY_ZH.md', 'assets': len(assets),
                      'brier': {k: v['aggregate']['brier'] for k, v in report['methods'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
