#!/usr/bin/env python3
"""CPU-only paper figures from saved development predictions; never loads models."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pathrel.parent_pilot_data import load_pilot
from pathrel.selective_risk import risk_at_coverage

OUT = Path(__file__).resolve().parent
DATA = ROOT / 'results/parent_group_pilot_v1/data'
PARENT = ROOT / 'results/parent_group_pilot_v1'
COHERENT = ROOT / 'results/coherent_parent_pilot_v1'
SEEDS = [20260910, 20260911, 20260912]
RADII = [0, 10, 20]
LABELS = {'coherent_categorical': 'ConPath coherent', 'correlated': 'ConPath 原采样器',
          'independent': '独立单元补全', 'deterministic': '确定性补全'}
COLORS = {'coherent_categorical': '#168b78', 'correlated': '#3972ab',
          'independent': '#cf813c', 'deterministic': '#68737f'}
SOURCE_HASHES = {}
INVENTORY = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path, expected=None):
    path = Path(path)
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'Source hash changed: {path}')
    SOURCE_HASHES[str(path.relative_to(ROOT))] = actual
    return path


def read(path, expected=None):
    return json.loads(bound(path, expected).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metric(p, y, w):
    w = w / w.sum(); clipped = np.clip(p, 1e-6, 1-1e-6)
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = []
    for i in range(10):
        mask = bins == i; mass = float(w[mask].sum())
        reliability.append({'bin': i, 'weight': mass, 'count': int(mask.sum()),
                            'confidence': float(np.average(p[mask], weights=w[mask])) if mass else None,
                            'accuracy': float(np.average(y[mask], weights=w[mask])) if mass else None})
    accepted = p >= .8; coverage = float(w[accepted].sum())
    return {'brier': float(np.sum(w * (p-y)**2)),
            'nll': float(-np.sum(w * (y*np.log(clipped)+(1-y)*np.log1p(-clipped)))),
            'ece': float(sum(b['weight'] * abs(b['confidence']-b['accuracy']) for b in reliability if b['weight'])),
            'false_safe_at_0_8': float(np.sum(w[accepted]*(1-y[accepted])) / coverage) if coverage else None,
            'coverage_at_0_8': coverage, 'risk30': risk_at_coverage(p, y, w, .3)['false_safe_rate'],
            'reliability': reliability}


def load_runs(samples):
    lookup = {}
    for sample in samples:
        for qi, candidate in enumerate(sample.candidate_indices):
            for ri, radius in enumerate(RADII):
                lookup[(sample.row['global_id'], int(candidate), radius)] = (
                    sample.row['parent_group'], bool(sample.targets[qi, ri]), 1/(40 * sample.targets.size))
    runs = {}
    for method in LABELS:
        for seed in (SEEDS[:2] if method == 'coherent_categorical' else SEEDS):
            folder = COHERENT / 'evaluation' / str(seed) if method == 'coherent_categorical' else PARENT / 'evaluation' / method / str(seed)
            report = read(folder / 'report.json')
            k = 1 if method == 'deterministic' else 32
            path = bound(folder / f'predictions_k{k}.csv', report['files'][f'predictions_k{k}.csv'])
            rows = list(csv.DictReader(path.open()))
            values = {}
            for row in rows:
                key = row['global_id'], int(row['candidate_index']), int(row['radius_cells'])
                if key in values or key not in lookup or lookup[key][0] != row['parent_group']:
                    raise ValueError('Predictions changed the frozen query identity')
                values[key] = float(row['probability'])
            if set(values) != set(lookup):
                raise ValueError('Incomplete validation predictions')
            keys = list(lookup)
            p = np.array([values[key] for key in keys]); y = np.array([lookup[key][1] for key in keys], float)
            w = np.array([lookup[key][2] for key in keys])
            scores = metric(p, y, w)
            for key in ('brier', 'risk30'):
                if not np.isclose(scores[key], report['budgets'][str(k)][key], atol=1e-12, rtol=0):
                    raise ValueError(f'Saved aggregate does not reproduce: {method}/{seed}/{key}')
            by_radius = {}
            for radius in RADII:
                keep = np.array([key[2] == radius for key in keys])
                by_radius[str(radius)] = metric(p[keep], y[keep], w[keep])
            runs[(method, seed)] = {'method': method, 'seed': seed, 'K': k, 'path': str(path.relative_to(ROOT)),
                                   'report_path': str((folder/'report.json').relative_to(ROOT)),
                                   'metrics': scores, 'by_radius': by_radius, 'map': report['budgets'][str(k)]['map'],
                                   'probabilities': p, 'targets': y, 'weights': w, 'values': values,
                                   'keys': keys, 'folder': folder, 'report': report}
    return runs


def save(fig, folder, name, caption_zh, caption_en, *, evidence, scope, details=None):
    fig.patch.set_facecolor('white')
    files = []
    for suffix in ('svg', 'png', 'pdf'):
        path = folder / f'{name}.{suffix}'
        fig.savefig(path, dpi=170, bbox_inches='tight')
        files.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    plt.close(fig)
    INVENTORY.append({'name': name, 'caption_zh': caption_zh, 'caption_en': caption_en,
                      'scope': scope, 'source_bundle_sha256': evidence, 'files': files, **(details or {})})


def footnote(fig, text, evidence):
    fig.text(.5, -.01, text + f'\n来源清单 SHA256={evidence[:16]}；完整来源见同目录 manifest.json',
             ha='center', va='top', fontsize=8, color='#52616a')


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15, axis='y')
    ax.set_axisbelow(True)


def groups():
    return [('主比较：原ConPath与基线，三个种子', SEEDS, ['correlated', 'independent', 'deterministic']),
            ('补充：coherent匹配比较，两个种子', SEEDS[:2], list(LABELS))]


def numerical_figures(runs, folder, evidence):
    common = 'validation-only 开发比较；40父级地点；515冻结查询×r=0/10/20格；K=32，确定性实际K=1'
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, (title, seeds, methods) in enumerate(groups()):
        for col, (key, label) in enumerate([('brier','Event Brier ↓'),('nll','Event NLL ↓'),('ece','Event ECE ↓')]):
            ax=axes[row,col]
            for mi, method in enumerate(methods):
                values=np.array([runs[(method,s)]['metrics'][key] for s in seeds])
                ax.errorbar(mi,values.mean(),yerr=values.std(ddof=1),fmt='o',color=COLORS[method],capsize=4)
                ax.scatter(mi+np.linspace(-.09,.09,len(seeds)),values,color=COLORS[method],s=22,alpha=.55)
            ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=15,ha='right',fontsize=8)
            ax.set_ylabel(label); ax.set_title(title if col==1 else '',fontsize=10); tidy(ax)
    fig.suptitle('同一队列内的任务指标：重复数分组，不混成同一次实验',fontsize=16)
    fig.tight_layout(rect=(0,.04,1,.95))
    footnote(fig,common+'\n上排seed=20260910/20260911/20260912；下排seed=20260910/20260911；误差线=种子标准差，不是置信区间。NLL/ECE为保存预测的事后描述，未重新选点。',evidence)
    save(fig,folder,'03_event_metrics','父级地点等权的事件指标，分开呈现两种子coherent比较和三种子原采样器比较。',
         'Event metrics on the same 40-parent development cohort. Two-seed coherent comparisons and three-seed original-sampler comparisons are shown separately; error bars are training-seed SD, not confidence intervals.',evidence=evidence,scope='validation-only development')

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,seed in zip(axes,SEEDS):
        ax.plot([0,1],[0,1],'--',color='#9ca6aa',lw=1,label='理想一致线')
        for method in ['correlated','independent','deterministic']:
            bins=[b for b in runs[(method,seed)]['metrics']['reliability'] if b['weight']]
            ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=LABELS[method],color=COLORS[method],ms=4)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='原始可达事件频率',ylabel='实际可达比例',title=f'seed={seed}')
        tidy(ax)
    axes[2].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    fig.suptitle('主比较可靠性：固定10分箱，空箱留空；三个种子分别显示',fontsize=15);fig.tight_layout(rect=(0,.035,1,.93))
    footnote(fig,common+'；seed=20260910/20260911/20260912。无Platt或验证集调温度；连线仅帮助读图，不是连续校准保证。',evidence)
    save(fig,folder,'04_reliability','原ConPath及两个基线三个匹配种子的原始事件可靠性；分箱内按父级地点等权计算。',
         'Uncalibrated event reliability for the original ConPath and baselines, for each of the three matched training seeds, using ten fixed bins and equal-parent weights. Empty bins are omitted; line segments are visual guides.',evidence=evidence,scope='validation-only development')

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    coverage=np.linspace(.01,1,100)
    for ax,(title,seeds,methods) in zip(axes,groups()):
        for method in methods:
            curves=[]
            for seed in seeds:
                run=runs[(method,seed)]
                curve=np.array([risk_at_coverage(run['probabilities'],run['targets'],run['weights'],c)['false_safe_rate'] for c in coverage])
                curves.append(curve);ax.plot(coverage*100,curve*100,color=COLORS[method],alpha=.25,lw=.8)
            ax.plot(coverage*100,np.mean(curves,0)*100,color=COLORS[method],label=LABELS[method],lw=2)
        ax.axvline(30,color='#99a3a5',ls=':',lw=1)
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='接受查询比例 / %',ylabel='接受查询中实际不可达 / %',title=title);tidy(ax)
    axes[1].legend(fontsize=8);fig.suptitle('风险—覆盖率：保留所有方向与失败，不按标签拆同分查询',fontsize=15);fig.tight_layout(rect=(0,.035,1,.93))
    footnote(fig,common+'\n左seed=20260910/11/12；右seed=20260910/11。细线为各种子，粗线为其均值；同分边界按相同比例接受，仅描述期望风险。',evidence)
    save(fig,folder,'05_risk_coverage','相同覆盖率下的误判风险；不把较低覆盖率下的风险当作直接安全优势。',
         'Selective risk versus weighted coverage, with label-independent fractional acceptance of tied boundary scores. Thin curves show individual seeds and thick curves their means; no deployment safety guarantee is implied.',evidence=evidence,scope='validation-only development')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(title,seeds,methods) in zip(axes,groups()):
        for method in methods:
            values=np.array([[runs[(method,s)]['by_radius'][str(r)]['brier'] for r in RADII] for s in seeds])
            ax.errorbar(RADII,values.mean(0),yerr=values.std(0,ddof=1),marker='o',color=COLORS[method],label=LABELS[method],capsize=3)
        ax.set(xticks=RADII,xlabel='机器人半径 / grid cell',ylabel='Event Brier ↓',title=title);tidy(ax)
    axes[1].legend(fontsize=8);fig.suptitle('不同足迹半径：各半径重新按40个父级地点等权',fontsize=15);fig.tight_layout(rect=(0,.035,1,.93))
    footnote(fig,common+'\n左seed=20260910/11/12；右seed=20260910/11。半径不换算为米；误差线为种子标准差。',evidence)
    save(fig,folder,'06_radius','固定查询在0、10、20格圆盘足迹下的事件误差。',
         'Event Brier at the three frozen disk-footprint radii, in raster cells. Equal-parent weights are recomputed within each radius; error bars denote seed SD.',evidence=evidence,scope='validation-only development')

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,(title,seeds,methods) in enumerate(groups()):
        for col,(key,label) in enumerate([('sample_vote_cell_brier','隐藏地图频率 Brier ↓'),('mean_blocked_iou','平均障碍 IoU ↑')]):
            ax=axes[row,col]
            for i,method in enumerate(methods):
                values=np.array([runs[(method,s)]['map'][key] for s in seeds])
                ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='o',color=COLORS[method],capsize=4)
            ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=12,ha='right',fontsize=8)
            ax.set_ylabel(label);ax.set_title(title,fontsize=10);tidy(ax)
    fig.suptitle('地图指标：不能将每格可通行频率称为路径概率',fontsize=15);fig.tight_layout(rect=(0,.04,1,.94))
    footnote(fig,common+'\n上排seed=20260910/11/12；下排seed=20260910/11；地图分数仅隐藏有效单元，与半径无关。误差线为种子标准差。',evidence)
    save(fig,folder,'07_map_metrics','同一隐藏有效区域的地图指标；与路径事件评分区分。',
         'Hidden valid-cell map metrics on the same cohort, separated from path-event scores. The empirical cellwise free frequency is not a path probability.',evidence=evidence,scope='validation-only development')


def architecture(folder,evidence):
    fig,ax=plt.subplots(figsize=(17,10));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,title,body,color='#edf5f3'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#7e9691',lw=1.2))
        ax.text(x+w/2,y+h-.045,title,ha='center',va='top',fontsize=13,fontweight='bold')
        ax.text(x+w/2,y+h-.093,body,ha='center',va='top',fontsize=10,linespacing=1.55)
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,color='#436c65',lw=1.5,linestyle='--' if dashed else '-'))
    box(.025,.70,.21,.23,'共同观测与支持','3通道：可通行 / 阻挡 / 未知\n有效支持 S 单独约束\n不输入完整参考地图')
    box(.285,.70,.25,.23,'TinyBEVUNet 编码器','内部添加二维坐标平面\n16 → 32 → 64通道，下采样两次\n双线性上采样 + 跳连\n池化全局上下文相加')
    box(.585,.70,.39,.23,'随机地图头','μ：两类logit位置（不是边际概率）\nB：4维全局高斯因子；σ：局部噪声尺度\nL_k = μ + B z_k / √4 + σ · filter(ε_k)\n原版局部Gaussian核5×5')
    box(.585,.38,.39,.24,'最终方法：原版类别采样','逐单元独立Gumbel + straight-through\n空间相关来自前一框的Gaussian因子/局部logit\n补充coherent：9×9、σ=3 Gaussian copula\n仅补充诊断，非最终方法；温度0.7用于反向')
    box(.285,.38,.25,.24,'投影已知证据','保留已观测可通行与障碍\n支持范围外封闭为障碍\n保留每个真实世界\n不以软均值图代替采样世界')
    box(.025,.38,.21,.24,'K张二值世界','W_1, W_2, ..., W_K\n独立随机流生成世界\n逐格mean(W_k)是地图频率\n不会直接成为路径分数')
    box(.025,.075,.21,.20,'精确任务几何','固定S / G / 半径 r（格）\n圆盘足迹 + 四邻域\n超出地图视为障碍\n逐世界得事件 ρ_k ∈ {0,1}')
    box(.285,.075,.25,.20,'事件频率','p(S,G,r) = mean_k ρ_k\n不是预测一条路线\n不保证单张世界对应真实环境\n最终验证用完整精确连通性')
    box(.585,.075,.39,.20,'训练监督（仅训练时）','完整参考：隐藏图BCE + 0.1 variogram\n再加 2 × 事件Brier U-statistic\n硬事件前向由精确oracle修正\n反向保留最多256步松弛传播；不是精确梯度','#f7f0e6')
    for a,b in [((.236,.81),(.285,.81)),((.535,.81),(.585,.81)),((.78,.70),(.78,.62)),((.585,.5),(.535,.5)),((.285,.5),(.236,.5)),((.13,.38),(.13,.275)),((.236,.175),(.285,.175))]:arrow(a,b)
    arrow((.585,.175),(.535,.175),True)
    fig.suptitle('最终ConPath：相关Gaussian logit → 独立类别抽样 → 足迹事件',fontsize=18,y=.995)
    footnote(fig,'架构示意，不是实验结果；对应parent/coherent pilot实现，seed/scene/query/radius数值均不适用。\nno-global仅关闭解码器B z项，保留编码器全局上下文；no-event仅关闭事件训练项。coherent与原版均分别训练。',evidence)
    save(fig,folder,'01_architecture','按实际模型、解码器和父级地点训练脚本绘制；硬前向与松弛反向明确区分。',
         'Implemented ConPath pipeline. Correlated map samples are constrained by observed evidence and support, then evaluated by exact disk-footprint connectivity. The original and coherent categorical samplers are distinct trained variants. Exact hard event correction is paired with a bounded relaxed backward surrogate.',evidence=evidence,scope='implementation schematic; no experiment score')


def map_rgb(sample,world=None):
    palette=np.array([[52,69,82],[98,177,155],[220,230,240]],dtype=np.uint8)
    state=np.where(sample.observation[0]>.5,1,np.where(sample.hidden,2,0)) if world is None else world.astype(np.uint8)
    rgb=palette[state].copy(); rr,cc=np.indices(sample.valid.shape);outside=~sample.valid
    rgb[outside]=np.where((((rr[outside]//8)+(cc[outside]//8))%2)[:,None],[232,236,238],[248,249,247])
    return rgb


def qualitative(samples,gallery,runs,folder,evidence):
    byid={s.row['global_id']:s for s in samples}; worlds={}
    for method in ['coherent_categorical','correlated','independent']:
        run=runs[(method,SEEDS[0])];p=bound(run['folder']/'worlds.npz',run['report']['files']['worlds.npz'])
        with np.load(p,allow_pickle=False) as z:
            ids=z['global_ids'].tolist();values=z['worlds']
            if set(ids)!=set(byid) or values.shape!=(40,32,256,256) or values.dtype!=bool:raise ValueError('Saved worlds mismatch')
            worlds[method]={gid:values[ids.index(gid)] for gid in byid}
    findings=[]
    for index,case in enumerate(gallery['examples']):
        sample=byid[case['global_id']];q=np.flatnonzero(sample.candidate_indices==case['candidate_index'])
        if len(q)!=1:raise ValueError('Gallery query missing')
        qi=int(q[0]);radius=case['radius_cells'];ri=RADII.index(radius)
        if not np.array_equal(sample.starts[qi],case['start']) or not np.array_equal(sample.goals[qi],case['goal']):raise ValueError('Gallery endpoints changed')
        ref=sample.targets[qi].tolist()
        clearance=distance_transform_edt(np.pad(sample.target,1,constant_values=False))[1:-1,1:-1]
        endpoints_fit=all(clearance[tuple(p)]>radius for p in [sample.starts[qi],sample.goals[qi]])
        bottleneck=bool(ref[0] and not ref[ri] and endpoints_fit)
        findings.append({'global_id':case['global_id'],'query_id':f"{case['global_id']}:q{case['candidate_index']:03d}",
                         'radius_cells':radius,'reference_events_r0_r10_r20':ref,'endpoints_fit_radius':bool(endpoints_fit),
                         'reference_bottleneck_at_displayed_radius':bottleneck,'reference_unreachable':not bool(ref[ri])})
        fig,axes=plt.subplots(3,6,figsize=(22,15));row_labels=[]
        for row,method in enumerate(['correlated','independent','coherent_categorical']):
            values=worlds[method][case['global_id']]
            if values[:,~sample.valid].any() or np.any(values[:,~sample.hidden]!=(sample.observation[0,~sample.hidden]>.5)):
                raise ValueError('Saved worlds changed known evidence/support')
            prediction=runs[(method,SEEDS[0])]['values'][(case['global_id'],int(case['candidate_index']),radius)]
            for col in range(6):
                ax=axes[row,col];world=None if col==0 else sample.target if col==1 else values[col-2]
                ax.imshow(map_rgb(sample,world),interpolation='nearest')
                for point,color,marker,label in [(sample.starts[qi],'#287ab4','o','S'),(sample.goals[qi],'#c27425','D','G')]:
                    rr,cc=point;ax.scatter(cc,rr,c=color,marker=marker,s=28,edgecolors='white',linewidths=.6)
                    ax.annotate(label,(cc,rr),xytext=(4,-8),textcoords='offset points',fontsize=8,color=color,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':.3})
                    if radius:ax.add_patch(Circle((cc,rr),radius,fill=False,color=color,lw=.8))
                title='已观测输入' if col==0 else '完整参考' if col==1 else f'实际样本 {col-1}/32'
                ax.set_title(title,fontsize=10,pad=6);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
            row_labels.append(('补充候选：' if method=='coherent_categorical' else '主比较：')+LABELS[method]+f' · seed={SEEDS[0]} · 全部K32世界事件频率={prediction:.5f}')
        query_id=f"{case['global_id']}:q{case['candidate_index']:03d}"
        fig.suptitle(f"固定案例 {index+1}/10 · scene={case['scene']}\n{query_id} · 半径={radius} grid cell · seed={SEEDS[0]} · validation-only\n参考事件：{'可达' if ref[ri] else '不可达'}；仅固定展示保存顺序前4/32张，不按结果选世界",fontsize=13,y=.995)
        legend=[Patch(color='#62b19b',label='可通行'),Patch(color='#344552',label='障碍'),Patch(color='#dce6f0',label='输入未知'),Patch(color='#e8ecee',label='棋盘纹：支持外封闭')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.026),ncol=4,frameon=False,fontsize=10)
        fig.subplots_adjust(top=.90,bottom=.09,hspace=.25,wspace=.04,left=.075,right=.995)
        fig.canvas.draw()
        for row,label in enumerate(row_labels):
            fig.text(.535,max(ax.get_position().y1 for ax in axes[row])+.028,label,ha='center',fontsize=13,fontweight='bold',color='#35585a')
        footnote(fig,'S蓝圆=起点；G橙菱形=目标；空心圆=机器人半径；不画无依据连接线。\n统计来自全部K32；图只显示固定前4张。父级地点开发验证；coherent验证反馈已用于开发方向选择。',evidence)
        save(fig,folder,f'02_fixed_case_{index:02d}',f"固定哈希案例{index+1}，同一输入、参考及同一种子的前三方法真实世界。",
             f"Fixed pre-existing gallery case {index+1}, {query_id}, radius {radius} cells. All methods use the same observation and query. The first four saved worlds are shown in order; displayed event frequencies use all 32 worlds. Development validation only.",
             evidence=evidence,scope='validation-only development',details={'scene_id':case['scene'],'query_id':query_id,'radius_cells':radius,'seed':SEEDS[0],'displayed_world_indices':[0,1,2,3],'metric_K':32,'reference_geometry':findings[-1]})
    return findings


def synthetic(folder,evidence):
    paths=['p0_neural_cuda_contextplane_v4','p0_neural_cuda_contextplane_seed20260828_v4','p0_neural_cuda_contextplane_noreach_v4']
    reports=[read(ROOT/'results'/p/'report.json') for p in paths]
    labels=['完整模型 · 20260827','完整模型 · 20260828','无事件损失 · 20260827']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for i,(report,label,color) in enumerate(zip(reports,labels,['#168b78','#3972ab','#cf813c'])):
        axes[0].scatter(i,report['event_metrics']['brier'],color=color,s=65)
        axes[1].plot([0,1,2],[v['brier'] for v in report['event_metrics_by_radius']],'o-',label=label,color=color)
    axes[0].set_xticks(range(3),labels,rotation=12,ha='right',fontsize=9);axes[0].set_ylabel('Event Brier ↓')
    axes[1].set(xlabel='半径 / grid cell',ylabel='Event Brier ↓',xticks=[0,1,2]);axes[1].legend(fontsize=8)
    for ax in axes:tidy(ax)
    fig.suptitle('P0合成留出模板：仅保留已有真实报告，不与FlatLands并表',fontsize=15);fig.tight_layout(rect=(0,.05,1,.93))
    footnote(fig,'synthetic-heldout；报告记录test_templates=4、worlds_per_template=24；1152事件/报告；r=0/1/2格；K=128。\n无事件损失仅一个匹配seed；另一个完整seed单独列，不伪称三次重复。无地图数组，因此不生成P0模型定性图。',evidence)
    save(fig,folder,'08_synthetic_event_ablation','P0已保存报告中的事件指标；两种子完整模型与一个匹配种子的无事件损失控制分别标识，不作跨数据集并表。',
         'Existing synthetic held-out-template diagnostics, reported separately from FlatLands. The no-event-loss control has one matched seed; the additional full-model seed is shown separately, not treated as an extra ablation repeat.',evidence=evidence,scope='synthetic-heldout diagnostic')
    return [{'path':str((ROOT/'results'/p/'report.json').relative_to(ROOT)),'seed':r['protocol']['seed'],'metrics':r['event_metrics'],'by_radius':r['event_metrics_by_radius']} for p,r in zip(paths,reports)]


def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
                         'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
    parent_protocol=read(DATA/'protocol.json');coherent_protocol=read(COHERENT/'protocol.json');seal=read(DATA/'seal.json')
    if parent_protocol['physical_archive_split']!='train' or coherent_protocol['data_seal_sha256']!=digest(DATA/'seal.json'):
        raise ValueError('Unsafe or mismatched development data')
    for name in ['protocol.json','selected.csv','data_audit.json','input_queries_before_target.jsonl']:bound(DATA/name,seal['files'][name])
    audit=read(DATA/'data_audit.json')
    if not audit['passed'] or audit['parent_overlap'] or audit['new_physical_test_images_opened']:raise ValueError('Data audit not eligible')
    gallery=read(ROOT/'site/data/coherent_pilot_gallery_zh.json')
    verification=read(COHERENT/'gallery_verification.json')
    if not verification['passed'] or verification['gallery_sha256']!=digest(ROOT/'site/data/coherent_pilot_gallery_zh.json'):raise ValueError('Gallery receipt mismatch')
    for method_root in [PARENT,COHERENT]:
        read(method_root/'analysis.json');read(method_root/'verification.json')
    for name in ['src/pathrel/model.py','src/pathrel/stochastic_decoder.py','src/pathrel/coherent_categorical.py','src/pathrel/reachability.py','src/pathrel/losses.py','src/pathrel/parent_pilot_data.py','src/pathrel/selective_risk.py','scripts/train_parent_group_pilot.py']:
        bound(ROOT/name)
    for name in ['p0_neural_cuda_contextplane_v4','p0_neural_cuda_contextplane_seed20260828_v4','p0_neural_cuda_contextplane_noreach_v4']:bound(ROOT/'results'/name/'report.json')
    samples=load_pilot(DATA,'validation')
    if len(samples)!=40 or sum(s.targets.size for s in samples)!=1545:raise ValueError('Wrong cohort')
    for sample in samples:bound(DATA/'packets'/f"{sample.row['global_id']}.npz",seal['files'][f"packets/{sample.row['global_id']}.npz"])
    if len(gallery['examples'])!=10 or any(c['global_id'] not in {s.row['global_id'] for s in samples} for c in gallery['examples']):raise ValueError('Gallery cohort mismatch')
    runs=load_runs(samples)
    for method in ['coherent_categorical','correlated','independent']:
        run=runs[(method,SEEDS[0])];bound(run['folder']/'worlds.npz',run['report']['files']['worlds.npz'])
    bound(Path(__file__))
    evidence=hashlib.sha256(json.dumps(SOURCE_HASHES,sort_keys=True).encode()).hexdigest()
    folder=OUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');folder.mkdir(exist_ok=False)
    write_json(folder/'selection_before_rendering.json',{'source_gallery':str((ROOT/'site/data/coherent_pilot_gallery_zh.json').relative_to(ROOT)),
               'gallery_sha256':digest(ROOT/'site/data/coherent_pilot_gallery_zh.json'),'examples':gallery['examples'],
               'selection':'Retain all 10 pre-existing hash-selected cases in recorded order; display fixed world indices 0..3 of saved K32.',
               'new_outcome_based_selection':False,'new_inference':False,'new_test_asset_reads':0})
    architecture(folder,evidence);numerical_figures(runs,folder,evidence)
    geometry=qualitative(samples,gallery,runs,folder,evidence);p0=synthetic(folder,evidence)
    serial=[]
    for run in runs.values():
        serial.append({key:run[key] for key in ['method','seed','K','path','report_path','metrics','by_radius','map']})
    write_json(folder/'metrics_from_saved_predictions.json',{'cohort':'parent_group_pilot_v1','validation_parents':40,'events':1545,'runs':serial,
               'weighting':'equal parent, then equal frozen events within each parent','brier_risk30_reproduced_existing_reports':True,
               'nll_ece_status':'post-hoc descriptive recomputation from unchanged saved predictions, not new model evaluation',
               'nll_epsilon':1e-6,'ece_bins':10,'no_calibration_fitted':True})
    missing=[{'requested':'父级隔离no-event/no-global/direct定性对照','status':'不可生成','reason':'该cohort没有这些方法的已保存训练/世界结果；旧K128属于含地点重叠及历史physical-test访问的不同队列，不能补入。'},
             {'requested':'P0 obs/reference/模型多世界定性','status':'不可生成','reason':'三个P0目录仅有报告、日志和模型checkpoint，未发现已保存地图数组；本任务禁止新模型推理。'},
             {'requested':'历史K128两例作为无选择偏差的消融主图','status':'不采用','reason':'旧例子名称与既有选择记录含positive_recovery/false_safe_avoided，按结果挑选；不能把沿用旧挑图称为盲选。'},
             {'requested':'严格窄瓶颈定性','status':'已有固定案例中可用' if any(g['reference_bottleneck_at_displayed_radius'] for g in geometry) else '不可生成独立瓶颈案例',
              'reason':'只在既有10个固定查询中核对reference r0可达、当前半径不可达且起终点均放得下圆盘；不另按模型结果选图。',
              'cases':[g for g in geometry if g['reference_bottleneck_at_displayed_radius']]}]
    manifest={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Existing evidence only; external training cancelled; no new experiments',
              'cohort':{'train_parents':100,'calibration_parents':25,'validation_parents':40,'queries':515,'events':1545,'radii_cells':RADII,
                        'baseline_seeds':SEEDS,'coherent_matched_seeds':SEEDS[:2],'validation_reused_for_development':True,'final_test':False},
              'protocols':{'parent':{'path':str((DATA/'protocol.json').relative_to(ROOT)),'sha256':digest(DATA/'protocol.json')},
                           'coherent':{'path':str((COHERENT/'protocol.json').relative_to(ROOT)),'sha256':digest(COHERENT/'protocol.json')}},
              'source_bundle_sha256':evidence,'source_hashes':SOURCE_HASHES,'figures':INVENTORY,'reference_geometry_for_fixed_cases':geometry,
              'missing_or_excluded':missing,'new_inference':False,'new_training':False,'gpu_used':False,'new_test_asset_reads':0,
              'external_model_images_included':False,'formal_superiority_claim':False}
    write_json(folder/'manifest.json',manifest)
    write_json(OUT/'current.json',{'snapshot':str(folder.relative_to(ROOT)),'manifest_sha256':digest(folder/'manifest.json')})
    lines=['# 现有证据的论文图片索引','',
      '外部训练已取消。本批仅重绘已有数值与保存世界，没有启动模型推理、训练或读取测试资产；不把外部短训练图放进论文主图。',
      '',f"不可变快照：[{folder.name}]({folder.relative_to(ROOT)}/manifest.json)。每幅图均有SVG、PNG、PDF；完整源文件SHA256、协议、种子与半径见manifest。",'',
      '主数据为100/25/40父级地点开发队列（40验证地点、515端点查询、1545半径事件）。原ConPath/independent/deterministic三个种子与coherent两个种子分组展示；验证反馈已用于开发，不能写成最终测试或外部方法优越性。',
      '', '## 已生成图片','']
    for figure in INVENTORY:
        png=next(f['path'] for f in figure['files'] if f['path'].endswith('.png'))
        lines.extend([f"### {figure['name']}",'',f"[PNG]({png}) · [SVG]({png[:-3]}svg) · [PDF]({png[:-3]}pdf)",'',figure['caption_zh'],'',f"English caption: {figure['caption_en']}",''])
    lines+=['## 不可生成或不采用的部分','']
    for item in missing:lines.append(f"- **{item['requested']}：{item['status']}。** {item['reason']}")
    lines+=['','不可达案例已保留在原10张固定图中，均标参考事件；绿色连通不等于半径圆盘能通过。没有为凑正例/负例而替换固定查询。',
      '', '架构图为代码路径示意，不含新实验成绩。原版与coherent均重新训练，independent同时移除全局因素并缩小局部核；这些比较不能作为同边际纯相关性的因果证明。',
      '', '图中NLL/ECE是对已有CSV的事后描述性汇总；既有Brier和30%风险已逐run核对重现。没有在验证集拟合校准映射，也没有改变选中checkpoint。','']
    (ROOT/'PAPER_FIGURES.md').write_text('\n'.join(lines))
    print(json.dumps({'snapshot':str(folder.relative_to(ROOT)),'figures':len(INVENTORY),'files':sum(len(f['files']) for f in INVENTORY),'missing':missing},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
