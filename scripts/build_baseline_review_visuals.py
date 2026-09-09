#!/usr/bin/env python3
"""Chinese diagnostic plot, with seed SD explicitly distinct from place CIs."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
LABELS = {'correlated': 'ConPath', 'independent': '独立单元模型', 'tiny_deterministic': '确定性补全网络',
          'all_floor': '未知全部可通行', 'nearest_observed': '最近观测填充', 'all_blocked': '未知全部阻挡'}


def main():
    analysis_path = SITE / 'data/current_baseline_k4_analysis.json'
    parent_path = SITE / 'data/flatlands_parent_group_audit.json'
    analysis = json.loads(analysis_path.read_text())
    parent = json.loads(parent_path.read_text())
    history = json.loads((ROOT / 'results/unscenes_history_feasibility_v1/report.json').read_text())
    verification = json.loads((ROOT / 'results/baseline_review_20260909_v1/verification.json').read_text())
    access = json.loads((SITE / 'data/flatlands_read_scope_erratum.json').read_text())
    # The frozen numerical receipt is preserved. Derived public snapshots must
    # carry the later access correction even when rebuilt from old generators.
    analysis['physical_test_access_erratum'] = 'data/flatlands_read_scope_erratum.json'
    if 'test_evaluated' in analysis:
        analysis['test_evaluated_original_flag'] = analysis.pop('test_evaluated')
    analysis['provenance_test_model_evaluation_performed'] = False
    analysis['physical_test_observations_opened'] = 5
    if 'Parent-place isolation failed' not in analysis['bootstrap_scope']:
        analysis['bootstrap_scope'] += ' Parent-place isolation failed in27/160 observations. These are subscene-conditional diagnostics, not building-population intervals.'
    parent['candidate']['final_test_warning'] = 'Original physical ID test shares parents with development data. Earlier audits inspected32 ScanNet++ scenes and53 physical-test observations. No wholly-untouched-test claim; final eligibility requires full historical access audit and parent exclusions. Current parent audit itself is metadata-only.'
    parent['historical_access_erratum'] = 'data/flatlands_read_scope_erratum.json'
    for path, obj in [(analysis_path, analysis), (parent_path, parent)]:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    assert verification['passed'] and not parent['old_physical_place_isolation_passed']
    plt.rcParams.update({'font.family': FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False,
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#d6dfe1', 'text.color': '#20383c'})
    keys = list(LABELS)[:5]
    colors = ['#087f8c', '#bc7b37', '#667d9e', '#876c9c', '#7c8d86']
    assets = []
    folder = SITE / 'assets/zh/baseline-review'; folder.mkdir(exist_ok=True, parents=True)
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2, figsize=(4.8, 9.6) if mobile else (12.8, 4.6))
        for ax, metric, title, maximum in zip(np.ravel(axes), ['brier', 'risk30'],
                                            ['通路概率误差 Brier ↓', '接受30%查询后的误判率 ↓'], [.13, .165]):
            values = [analysis['methods'][k][metric]['mean'] for k in keys]
            deviations = [analysis['methods'][k][metric]['seed_sd'] or 0 for k in keys]
            ax.barh(range(5), values, color=colors, height=.51, alpha=.92)
            for i, (v, sd) in enumerate(zip(values, deviations)):
                if sd:
                    ax.errorbar(v, i, xerr=sd, color='#243e42', capsize=3, lw=1, fmt='none')
                label = f'{v:.4f}' if metric == 'brier' else f'{v*100:.2f}%'
                ax.text(v+sd+.003, i, label, va='center', fontsize=12 if mobile else 10)
            ax.set_yticks(range(5), [LABELS[k] for k in keys]); ax.invert_yaxis()
            ax.set_xlim(0, maximum + (.02 if mobile else 0)); ax.set_title(title, loc='left', pad=15, fontsize=14 if mobile else 13)
            ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0, pad=8)
            ax.tick_params(labelsize=13 if mobile else 11)
            if mobile:
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(4))
            if metric == 'risk30':
                from matplotlib.ticker import PercentFormatter
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        title = '旧队列诊断\n随机方法K=4，确定性方法K=1' if mobile else '旧队列诊断 · 随机方法K=4，确定性方法K=1'
        fig.suptitle(title, fontsize=14, y=.985, x=.02, ha='left')
        note = '横线：三次训练的标准差；规则无训练波动。\n27/160条验证观测与训练共享地点；\n不是新建筑泛化或论文排名。' if mobile else '横线：三次训练的标准差；规则无训练波动。\n27/160条验证观测与训练共享地点；不是新建筑泛化或论文排名。'
        fig.text(.02, .01, note, fontsize=10, color='#586b6e')
        fig.tight_layout(rect=(0, .09 if mobile else .12, .99, .92 if mobile else .95))
        stem = 'current-controls' + ('-mobile' if mobile else '')
        for ext in ('svg', 'pdf'):
            path = folder / f'{stem}.{ext}'
            fig.savefig(path, facecolor='white', metadata={'Creator': 'ConPath observed-result plot'})
            if ext == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            assets.append({'path': str(path.relative_to(SITE)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)
    result = {'created_for': '2026-09-09 baseline and parent audit', 'validation_only': True,
              'physical_test_observations_opened_in_current_replay': access['current_k4_replay_physical_test_observations_opened'],
              'provenance_test_model_evaluation_performed': False, 'formal_test_completed': False,
              'physical_test_untouched_claim_withdrawn': True, 'read_scope_erratum': 'data/flatlands_read_scope_erratum.json',
              'training_performed': False, 'external_superiority_established': False,
              'old_parent_isolation_passed': False, 'old_affected_observations': 27,
              'old_bootstrap_scope_correction': 'Subscene-conditional diagnostics only; old validation used for checkpoint selection and parent places overlap training. Not independent-building confidence intervals.',
              'labels': LABELS, 'assets': assets, 'history_train_summary': history['summary'],
              'verification': {**verification, 'test_images_opened': True, 'access_flag_corrected_by': 'data/flatlands_read_scope_erratum.json'},
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [SITE / 'data/current_baseline_k4_analysis.json', SITE / 'data/flatlands_parent_group_audit.json',
                                ROOT / 'results/unscenes_history_feasibility_v1/report.json', Path(__file__)]}}
    (SITE / 'data/baseline_review_zh.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(f'Built {len(assets)} SVG/PDF assets and public scientific-status snapshot.')


if __name__ == '__main__':
    main()
