#!/usr/bin/env python3
"""Chinese figures for external engineering checks, separate from paper comparisons."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('results/flatlands_external_publication_20260909')
    root.mkdir(parents=True, exist_ok=True)
    out = Path('site/assets/zh/flatlands-external'); out.mkdir(parents=True, exist_ok=True)
    profiles, batches, predictions, source_hashes = {}, {}, {}, {}
    for method in ['lama', 'flow']:
        p = Path(f'results/flatlands_{method}_profile_v1')
        profiles[method] = json.loads((p / 'report.json').read_text())
        for name, r in profiles[method]['outputs'].items(): assert sha(p / name) == r['sha256']
        predictions[method] = np.load(p / 'predictions.npz')
        b = Path(f'results/flatlands_{method}_batch64_v1')
        batches[method] = json.loads((b / 'report.json').read_text())
        for name, r in batches[method]['outputs'].items(): assert sha(b / name) == r['sha256']
        for f in [p / 'report.json', p / 'predictions.npz', b / 'report.json', b / 'training_trace.json']:
            source_hashes[str(f)] = sha(f)
    data = np.load('results/flatlands_lama_profile_v1/fixed_train_inputs.npz')
    rows = list(csv.DictReader(open('results/flatlands_lama_profile_v1/fixed_train_observations.csv')))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                         'font.size': 11, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    colors = ['#263447', '#c8eee2', '#dce1e8', '#efede5']
    categorical = ListedColormap(colors)
    scale = LinearSegmentedColormap.from_list('value', ['#f5f4eb', '#197f70'])
    scale.set_bad(colors[3])
    cases, assets = [], []
    # Literal first two frozen observations: no selection by targets or model scores.
    for i in [0, 1]:
        c = data['condition'][i]
        observed = np.where(c[2] == 0, 3, np.where(c[1] != 0, 2, c[0])).astype(int)
        target = np.where(c[2] == 0, 3, data['target'][i, 0]).astype(int)
        panels = {'observed': (observed, '① 模型输入 · 部分观测', False),
                  'lama': (predictions['lama'][f'values_{i}'][0], '② LaMa · 可通行输出值', True),
                  'flow': (predictions['flow'][f'worlds_{i}'].mean(0), '③ 流匹配 · 四图可通行比例', True),
                  'reference': (target, '④ 完整参考 · 仅用于核对', False)}
        for k, world in enumerate(predictions['flow'][f'worlds_{i}']):
            panels[f'flow_world{k}'] = (np.where(c[2] == 0, 3, world), f'流匹配 · 第{k+1}张完整采样', False)
        paths = {}
        for name, (array, title, continuous) in panels.items():
            fig, ax = plt.subplots(figsize=(4.6, 5.55), dpi=130)
            fig.subplots_adjust(left=.04, right=.96, bottom=.255, top=.84)
            ax.set_title(title, fontsize=13, pad=10)
            array = np.ma.masked_where(c[2] == 0, array) if continuous else array
            im = ax.imshow(array, cmap=scale if continuous else categorical, vmin=0, vmax=1 if continuous else 3, interpolation='nearest')
            ax.set_axis_off()
            fig.text(.5, .95, f"{rows[i]['source_dataset']} · {rows[i]['global_id']}", ha='center', fontsize=11, color='#5b6573')
            if continuous:
                cax = fig.add_axes([.15, .185, .7, .026])
                fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .25, .5, .75, 1])
                fig.text(.5, .102, '灰米色 = 有效范围外；非通路概率', ha='center', fontsize=10)
            else:
                legend = [(1, '可通行'), (0, '阻挡'), (3, '范围外')]
                if name == 'observed': legend.insert(2, (2, '未知'))
                fig.legend(handles=[Patch(facecolor=colors[k], label=text) for k, text in legend],
                           loc='lower center', bbox_to_anchor=(.5, .11), ncol=2, frameon=False, fontsize=10)
            fig.text(.5, .04, '固定训练样本 · 短训练尚未收敛\n只用于检查接口，不是正式对比成绩', ha='center', fontsize=10, color='#5b6573')
            path = out / f'case{i}-{name}.png'; fig.savefig(path, facecolor='white'); plt.close(fig)
            paths[name] = path.relative_to('site').as_posix()
            assets.append({'path': paths[name], 'sha256': sha(path), 'source_index': i, 'panel': name})
        cases.append({'index': i, 'global_id': rows[i]['global_id'], 'source': rows[i]['source_dataset'], 'panels': paths,
                      'profile_batch': 2, 'training_updates_including_warmup': 110})
    for method in ['lama', 'flow']:
        trace = json.loads(Path(f'results/flatlands_{method}_batch64_v1/training_trace.json').read_text())
        metric = 'reconstruction' if method == 'lama' else 'velocity_mse'
        name = 'LaMa：隐藏地图重建误差' if method == 'lama' else '条件流匹配：速度预测误差'
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
        ax.plot([r['iteration'] for r in trace], [r['losses'][metric] for r in trace], color='#197f70', linewidth=1.6)
        ax.set_title(name, loc='left', fontsize=14, pad=15)
        ax.set_xlabel('优化器更新次数（前5次为测速预热）')
        ax.set_ylabel('L1误差' if method == 'lama' else 'MSE误差')
        ax.grid(axis='y', alpha=.18); ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(1, 105)
        fig.text(.5, .015, '固定32个训练观测反复使用；这不是验证曲线，也不证明泛化或收敛。', ha='center', fontsize=9, color='#5b6573')
        fig.tight_layout(rect=[0, .045, 1, 1])
        for ext in ['svg', 'pdf']:
            path = out / f'{method}-training-loss.{ext}'
            fig.savefig(path, facecolor='white'); assets.append({'path': path.relative_to('site').as_posix(), 'sha256': sha(path), 'kind': 'batch64_training_loss'})
        plt.close(fig)
    # Review all fixed observations, including poor outputs, before publication.
    for page in range(4):
        fig, axes = plt.subplots(8, 4, figsize=(12, 20), dpi=90)
        for pos, i in enumerate(range(page * 8, page * 8 + 8)):
            c = data['condition'][i]
            arrays = [np.where(c[2] == 0, 3, np.where(c[1] != 0, 2, c[0])),
                      predictions['lama'][f'values_{i}'][0], predictions['flow'][f'worlds_{i}'].mean(0),
                      np.where(c[2] == 0, 3, data['target'][i, 0])]
            for col, array in enumerate(arrays):
                continuous = col in [1, 2]
                axes[pos, col].imshow(np.ma.masked_where(c[2] == 0, array) if continuous else array,
                    cmap=scale if continuous else categorical, vmin=0, vmax=1 if continuous else 3)
                axes[pos, col].set_title(f"{rows[i]['global_id']} · {['观测', 'LaMa输出', '流匹配四图比例', '参考'][col]}", fontsize=10)
                axes[pos, col].set_axis_off()
        fig.tight_layout(); fig.savefig(root / f'contact_{page}.png'); plt.close(fig)
    snapshot = {'schema_version': 1, 'engineering_only': True, 'formal_comparison': False, 'test_evaluated': False,
                'profiles': profiles, 'batch64': batches, 'selection': 'Literal first two rows of the previously frozen32 train list, without target/score filtering.',
                'cases': cases, 'assets': assets, 'source_hashes': source_hashes, 'renderer_sha256': sha(Path(__file__)),
                'readme': 'Images from 110 updates at batch2; curves/cost from separate 105-update effective-batch64 profiles. Neither is a formal trained baseline or held-out result.'}
    Path('site/data/flatlands_external_engineering_zh.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'assets': len(assets), 'public_cases': len(cases), 'audit_cases': 32}))


if __name__ == '__main__': main()
