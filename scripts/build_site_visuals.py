#!/usr/bin/env python3
"""Build Chinese teaching figures from frozen reports and real train/validation data.

No training is launched. Checkpoint inference is limited to the two already
published FlatLands diagnostic cases. Gallery/sequence selection uses training
scene identifiers and timestamps only, never model scores or test records.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import sys
import textwrap

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
ASSETS = ROOT / 'site/assets/zh'
DATA = ROOT / 'site/data'
RAW = ROOT / 'data/raw/unscenes3d/raw_package/unscenes3d-mini_raw'
LABELS = ROOT / 'data/raw/unscenes3d/label_package/unscenes3d-mini_label'
SELECTION = ROOT / 'results/p1_flatlands_query_audit_bounded/selected_observations.csv'
QUERIES = ROOT / 'results/p1_flatlands_query_audit_bounded/queries.csv'
ARCHIVE = ROOT / 'data/raw/flatlands/FlatLands_final_dataset.zip'
MANIFEST = ROOT / 'results/unscenes3d_contract_manifest_ground_valid/manifest.json'
TEAL, AMBER, INK, MUTED = '#157f77', '#c78231', '#263544', '#667582'
FREE, BLOCKED, UNKNOWN, OUTSIDE = '#65b59d', '#364654', '#dce4ec', '#f7f8fa'
METHODS = {
    'conpath': 'ConPath（相关补全）', 'independent': '独立单元对照',
    'mean_map': 'ConPath 均值地图', 'independent_mean_map': '独立对照均值地图',
    'completion': '确定性地图补全', 'coordinate': '坐标查询对照',
    'direct_query': '直接预测对照（单种子）', 'radius_prior': '训练集半径先验',
    'marginal_shuffle': '打散空间结构（评估干预）',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rgb(color):
    return np.array([int(color[i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def categorical(input_bev, valid, target=None):
    out = np.empty((*valid.shape, 3), dtype=np.uint8)
    out[:] = rgb(OUTSIDE)
    if target is None:
        out[valid & (input_bev[2] > .5)] = rgb(UNKNOWN)
        out[valid & (input_bev[0] > .5)] = rgb(FREE)
        out[valid & (input_bev[1] > .5)] = rgb(BLOCKED)
    else:
        out[valid & ~target] = rgb(BLOCKED)
        out[valid & target] = rgb(FREE)
    # A light checker marks invalid support separately from unknown cells.
    yy, xx = np.indices(valid.shape)
    out[(~valid) & (((xx // 4 + yy // 4) % 2) == 0)] = rgb('#e8ecf0')
    return out


def probability(values, valid):
    p = np.clip(np.asarray(values), 0, 1)
    # Single continuous scale, used for every model probability map.
    low, high = rgb('#f1ecd4').astype(float), rgb('#157f77').astype(float)
    out = np.rint(low + p[..., None] * (high-low)).astype(np.uint8)
    yy, xx = np.indices(valid.shape)
    out[~valid] = rgb(OUTSIDE)
    out[(~valid) & (((xx // 4 + yy // 4) % 2) == 0)] = rgb('#e8ecf0')
    return out


def map_svg(name, pixels, title, *, subtitle='', points=None, ramp=False):
    """A standalone, labelled map. Endpoints are never joined by a line."""
    raw = io.BytesIO()
    Image.fromarray(pixels).save(raw, format='PNG')
    uri = 'data:image/png;base64,' + base64.b64encode(raw.getvalue()).decode()
    height = 452
    legend = '米白到青绿表示单元格可通行概率从零到一，棋盘格表示有效范围外。' if ramp else '绿色表示可通行，深灰表示阻挡，浅灰表示未知，棋盘格表示有效范围外。'
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="380" height="{height}" viewBox="0 0 380 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(subtitle)}。{legend}S 为起点，G 为目标，不画直连路径。</desc>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}</style>',
            '<rect width="380" height="452" fill="white"/>',
            f'<text x="20" y="26" fill="{INK}" font-size="20" font-weight="600">{html.escape(title)}</text>',
            f'<text x="20" y="48" fill="{MUTED}" font-size="11">{html.escape(subtitle)}</text>',
            f'<image x="20" y="60" width="340" height="340" href="{uri}" style="image-rendering:pixelated"/>']
    if points:
        for label, point, color, symbol in [('S', points[0], '#326cbd', 'circle'), ('G', points[1], '#a35328', 'diamond')]:
            x, y = 20+(point[1]+.5)*340/pixels.shape[1], 60+(point[0]+.5)*340/pixels.shape[0]
            if symbol == 'circle':
                body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
            else:
                body.append(f'<path d="M{x:.2f},{y-9:.2f} l9,9 l-9,9 l-9,-9 Z" fill="{color}" stroke="white" stroke-width="2"/>')
            tx, ty = min(326, max(24, x+10)), min(391, max(75, y-11))
            body.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="16" font-weight="700" stroke="white" stroke-width="4" paint-order="stroke" fill="{color}">{label}</text>')
    if ramp:
        body += ['<defs><linearGradient id="p"><stop stop-color="#f1ecd4"/><stop offset="1" stop-color="#157f77"/></linearGradient></defs>',
                 '<text x="20" y="419" font-size="11" fill="#667582">单元格可通行概率</text>',
                 '<rect x="136" y="409" width="174" height="9" rx="2" fill="url(#p)"/>',
                 '<text x="117" y="419" font-size="11">0</text><text x="318" y="419" font-size="11">1</text>']
    else:
        for x, color, label in [(20, FREE, '可通行'), (102, BLOCKED, '阻挡'), (172, UNKNOWN, '未知'), (245, '#e8ecf0', '范围外')]:
            body.append(f'<rect x="{x}" y="410" width="10" height="10" rx="2" fill="{color}"/><text x="{x+15}" y="420" font-size="11" fill="{MUTED}">{label}</text>')
    body.append('<text x="20" y="441" font-size="11" fill="#667582">'+('蓝色圆点 S：起点 · 棕色菱形 G：目标 · 不表示规划路线' if points else '俯视地图 · 棋盘格区域不参与通路判断')+'</text>')
    body.append('</svg>')
    path = ASSETS / f'{name}.svg'
    path.write_text('\n'.join(body) + '\n')
    return f'assets/zh/{name}.svg'


def flat_dataset(split):
    from pathrel.flatlands_data import FlatLandsReplayDataset
    return FlatLandsReplayDataset(ARCHIVE, SELECTION, QUERIES, split=split)


def make_gallery():
    from pathrel.unscenes3d import load_frame
    from scripts.render_unscenes3d_qualitative import _resolve_adapter
    gallery = {'flatlands': [], 'unscenes3d': [], 'sequence': []}
    dataset = flat_dataset('train')
    groups = defaultdict(list)
    for index, observation in enumerate(dataset.observations):
        groups[observation.source_dataset].append((observation.global_id, index))
    selected = [sorted(groups[source])[0][1] for source in sorted(groups)]
    selected.append(sorted(groups['ZInD'])[1][1])
    for index in selected:
        sample = dataset[index]; row = sample.observation
        assert row.provenance_split == 'train'
        base = 'flatlands_' + row.global_id
        gallery['flatlands'].append({
            'id': row.global_id, 'source': row.source_dataset, 'scene': row.scene_id,
            'split': 'train', 'resolution_m': row.resolution,
            'observed': map_svg(base+'_observed', categorical(sample.input_bev, sample.epistemic_mask), '已观测地图', subtitle=row.source_dataset+' · '+row.global_id),
            'reference': map_svg(base+'_reference', categorical(sample.input_bev, sample.epistemic_mask, sample.target_free), '完整参考地图', subtitle='数据集标注 · '+row.global_id),
            'raw_members': {name: hashlib.sha256(dataset._zip().read(f'{row.packet_directory}/{name}')).hexdigest() for name in ('observed_floor.png','floor_map.png','unobserved.png','epistemic_mask.png','metadata.json')},
        })
    dataset.close()
    manifest = read(MANIFEST); adapter = _resolve_adapter(manifest)
    scenes = defaultdict(list)
    for record in manifest['records']['train']:
        assert record['location'] != 'location_6'
        scenes[record['scene_id']].append(record)
    by_location = defaultdict(list)
    for scene in sorted(scenes):
        by_location[scenes[scene][0]['location']].append(scene)
    # Two scenes per available training location, selected solely by scene id.
    scene_ids = [scene for location in sorted(by_location) for scene in by_location[location][:2]]
    def render(record):
        timestamp = record['timestamp']; base = 'unscenes_' + timestamp.replace('.', '_')
        frame = load_frame(timestamp, raw_root=RAW, label_root=LABELS, **adapter)
        camera = RAW / 'images' / f'{timestamp}.jpg'
        target = ASSETS / f'{base}_camera.jpg'
        shutil.copyfile(camera, target)  # Keep the released camera bytes unchanged.
        observed = map_svg(base+'_observed', categorical(frame.input_bev, frame.target_valid), '激光雷达观测地图', subtitle=record['scene_id']+' · 模型输入')
        reference = map_svg(base+'_reference', categorical(frame.input_bev, frame.target_valid, frame.target_free), '完整参考地图', subtitle='官方占用标注的保守地面投影')
        return {'id':timestamp, 'timestamp':timestamp, 'scene':record['scene_id'], 'location':record['location'], 'split':'train', 'camera':f'assets/zh/{target.name}', 'observed':observed, 'reference':reference, 'raw_sources':{str(p.relative_to(ROOT)):sha(p) for p in (camera, RAW/'clouds'/f'{timestamp}.bin', LABELS/'occ'/f'{timestamp}.npy')}}
    for scene in scene_ids:
        gallery['unscenes3d'].append(render(sorted(scenes[scene], key=lambda r:r['timestamp'])[0]))
    first_scene = scene_ids[0]
    for record in sorted(scenes[first_scene], key=lambda r:r['timestamp'])[:18]:
        gallery['sequence'].append(render(record))
    gallery.update(selection='FlatLands: first training observation per source plus the second ZInD observation. UnScenes3D: first two scene ids per training location, first timestamp per scene. Sequence: first 18 frozen training frames of the first scene. No outcome-based selection.', test_evaluated=False, camera_images_modified=False, sequence_has_paths=False)
    return gallery


def make_examples(device):
    import torch
    from scripts.render_flatlands_k128_advantage import _load_checkpoint_model, _posterior_visuals, _event
    metadata = read(DATA/'flatlands_k128_clean_candidate.json')
    dataset = flat_dataset('validation')
    by_id = {r.global_id:i for i,r in enumerate(dataset.observations)}
    models = {}; checkpoints = {}
    for variant, suffix in [('correlated','conpath'), ('independent','independent')]:
        path = ROOT / f'results/p1_flatlands_{suffix}_k128_support_clamped_v1/seed20260831_{suffix}/best.pt'
        models[variant], _ = _load_checkpoint_model(path, variant, torch.device(device))
        checkpoints[variant] = {'path':str(path.relative_to(ROOT)), 'sha256':sha(path)}
    output = []
    for case in metadata['cases']:
        sample = dataset[by_id[case['global_id']]]
        query = next(q for q in sample.retained_queries if q.candidate_index == case['candidate_index'])
        points = [(query.start_row, query.start_col), (query.goal_row, query.goal_col)]
        assert _event(sample.target_free, *points, case['radius_cells']) == bool(case['target'])
        base = 'example_'+case['case_id']
        record = {'id':case['case_id'], 'global_id':case['global_id'], 'source':case['source_dataset'], 'scene':case['scene_id'], 'split':'validation', 'candidate_index':case['candidate_index'], 'radius_cells':case['radius_cells'], 'radius_m':case['radius_cells']*sample.observation.resolution, 'target':case['target'], 'start':list(points[0]), 'goal':list(points[1]), 'selection_rule':case['selection_rule'], 'seed':20260831, 'visual_sampling_seed':case['visual_sampling_seed'], 'checkpoints':checkpoints, 'event_probability':{v:case['event_probability'][v]['20260831'] for v in models}, 'panels':{}}
        record['panels']['observed'] = map_svg(base+'_observed', categorical(sample.input_bev,sample.epistemic_mask), '① 机器人已经看到什么', subtitle='绿色已知可通行 · 浅灰尚未观测', points=points)
        record['panels']['reference'] = map_svg(base+'_reference', categorical(sample.input_bev,sample.epistemic_mask,sample.target_free), '④ 数据集给出的完整地图', subtitle='用于核对结果，不作为未知区域输入', points=points)
        for variant, model in models.items():
            visual = _posterior_visuals(model,sample,variant=variant,samples=128,sample_chunk=32,sampling_seed=case['visual_sampling_seed'],device=torch.device(device))
            record['panels'][variant] = map_svg(base+'_'+variant, probability(visual['posterior_mean'],sample.epistemic_mask), '② '+('ConPath 推测' if variant=='correlated' else '独立对照推测'), subtitle='128 次采样得到的单元格平均概率', points=points, ramp=True)
            world = visual['sample_worlds'][0]
            record['panels'][variant+'_sample'] = map_svg(base+'_'+variant+'_sample', categorical(sample.input_bev,sample.epistemic_mask,world), '③ 一次可能的完整世界', subtitle='固定展示第 1 次采样，不按好坏挑选', points=points)
            record.setdefault('displayed_world_event',{})[variant] = _event(world,*points,case['radius_cells'])
        output.append(record)
    dataset.close()
    return output


def make_charts(mobile=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    family = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams.update({'font.family':family,'font.size':13 if mobile else 12,'axes.unicode_minus':False,'axes.labelcolor':MUTED,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,'svg.fonttype':'none','svg.hashsalt':'conpath-zh-site-v1','lines.linewidth':2.3})
    analysis=read(DATA/'flatlands_clean_paper_analysis.json'); controls=read(DATA/'flatlands_clean_checkpoint_controls.json')['controls']
    chart_data={}
    def axis():
        fig,ax=plt.subplots(figsize=(5.1,6.2) if mobile else (10.4,5.8));fig.subplots_adjust(left=.20 if mobile else .12,right=.96,bottom=.17,top=.74 if mobile else .83)
        ax.set_axisbelow(True);ax.grid(axis='y',color='#e9edf0',linewidth=.8);ax.tick_params(length=0,pad=9)
        return fig,ax
    def save(fig,name,title,note,source):
        if mobile:
            title={'brier':'路径概率误差：越低越好','risk':'同样的覆盖率，比较误判风险','reliability':'预测概率，符合实际吗？','sampling':'增加采样后，误差改变多少？','dependence':'空间相关性有什么用？'}[name]
            note='\n'.join(textwrap.wrap(note,27))
            name += '-mobile'
        fig.text(.04,.965 if mobile else .955,title,ha='left',va='top',fontsize=16 if mobile else 19,weight='bold')
        fig.text(.04,.895,note,ha='left',va='top',fontsize=11,color=MUTED)
        for ext in ('svg','pdf'):
            path=ASSETS/f'{name}.{ext}'
            fig.savefig(path,metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None})
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig);chart_data[name]={'title':title,'note':note,'svg':f'assets/zh/{name}.svg','pdf':f'assets/zh/{name}.pdf','source':source}
    names=sorted((k for k in analysis['methods'] if k!='marginal_shuffle'),key=lambda k:analysis['methods'][k]['aggregate']['brier']['mean'])
    fig,ax=axis();fig.subplots_adjust(left=.36 if mobile else .30,right=.97 if mobile else .94,bottom=.14,top=.74 if mobile else .79)
    values=[analysis['methods'][k]['aggregate']['brier'] for k in names]
    colors=[TEAL if k=='conpath' else AMBER if k=='independent' else '#bcc8d0' for k in names]
    y=np.arange(len(names));means=np.array([v['mean'] for v in values]);sd=np.array([v['sample_sd'] or 0 for v in values])
    ax.barh(y,means,color=colors,height=.55)
    repeated=np.array([i for i,v in enumerate(values) if v['sample_sd'] is not None])
    ax.errorbar(means[repeated],y[repeated],xerr=sd[repeated],fmt='none',ecolor='#536574',capsize=3,elinewidth=1.2)
    for i,v in enumerate(values):ax.text(v['mean']+sd[i]+.003,i,f"{v['mean']:.4f}",va='center',fontsize=10 if mobile else 11)
    compact={'conpath':'ConPath','mean_map':'ConPath 均值图','independent_mean_map':'独立均值图','completion':'确定性补全','direct_query':'直接预测（单次）','coordinate':'坐标查询','independent':'独立单元对照','radius_prior':'半径先验（单次）'}
    ax.set(yticks=y,yticklabels=[compact[k] if mobile else METHODS[k] for k in names],xlabel='Brier 分数 ↓' if mobile else '路径事件 Brier 分数 ↓',xlim=(0,.215 if mobile else .195));ax.invert_yaxis();ax.grid(False);ax.grid(axis='x',color='#edf0f2')
    if mobile: ax.tick_params(axis='y',labelsize=11);ax.set_xticks([0,.1,.2])
    save(fig,'brier','路径概率的预测误差：越低越好','横线为三次训练的标准差；单种子与训练集先验不画误差线。','flatlands_clean_paper_analysis.json')
    fig,ax=axis()
    for method,color,marker in [('conpath',TEAL,'o'),('independent',AMBER,'s'),('mean_map','#738397','^')]:
        series=analysis['methods'][method]['equal_coverage'];x=np.array([float(k) for k in series]);ym=np.array([v['mean'] for v in series.values()]);ys=np.array([v['sample_sd'] or 0 for v in series.values()])
        ax.plot(x,ym,color=color,marker=marker,label=METHODS[method]);ax.fill_between(x,np.maximum(0,ym-ys),ym+ys,color=color,alpha=.075)
    ax.set(xlabel='接受查询的比例（覆盖率）',ylabel='误判通路比例 ↓' if mobile else '接受查询中的误判通路比例 ↓',xticks=[.1,.2,.3,.4,.5],xlim=(.08,.52),ylim=(0,.21));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.legend(frameon=False,loc='upper left',fontsize=10);ax.axvline(.3,color='#b7c3cb',linestyle=':',linewidth=1.2)
    save(fig,'risk','同样愿意接受多少查询，再比较风险','浅色带为训练种子标准差；30% 覆盖率下，两个模型的风险优势并不稳定。','flatlands_clean_paper_analysis.json')
    fig,ax=axis();ax.plot([0,1],[0,1],color='#aebac4',ls='--',lw=1.4,label='理想情况：预测与现实一致')
    for method,color,marker in [('conpath',TEAL,'o'),('independent',AMBER,'s')]:
        xs=[];ys=[]
        for i in range(10):
            bins=[s['metrics']['reliability'][i] for s in analysis['methods'][method]['seeds']];bins=[b for b in bins if b['weight']>0 and b['accuracy'] is not None]
            if bins:
                mass=sum(b['weight'] for b in bins);xs.append(sum(b['weight']*b['confidence'] for b in bins)/mass);ys.append(sum(b['weight']*b['accuracy'] for b in bins)/mass)
        ax.plot(xs,ys,color=color,marker=marker,label=METHODS[method])
    ax.set(xlabel='模型给出的通路概率',ylabel='对应查询实际有路的比例',xlim=(0,1),ylim=(0,1));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.legend(frameon=False,loc='upper left',fontsize=10)
    save(fig,'reliability','模型说“有 80% 把握”，能信多少？','折线越接近灰色对角线，预测概率越符合实际频率；十个概率分箱，按场景加权。','flatlands_clean_paper_analysis.json')
    fig,ax=axis()
    for variant,color,marker,label in [('correlated',TEAL,'o','ConPath'),('independent',AMBER,'s','独立单元对照')]:
        rows=[controls[variant]['event_brier'][str(k)] for k in [32,64,128]]
        ax.errorbar([32,64,128],[r['mean'] for r in rows],yerr=[r['sample_sd'] for r in rows],color=color,marker=marker,capsize=4,label=label)
    ax.set(xlabel='用于评估的地图采样数 K',ylabel='路径事件 Brier 分数 ↓',xticks=[32,64,128],ylim=(0,.12));ax.legend(frameon=False,loc='lower right')
    save(fig,'sampling','增加采样次数，结果改变多少？','固定已训练模型，32 / 64 / 128 使用同一随机流的前缀；误差线为种子标准差。','flatlands_clean_checkpoint_controls.json')
    fig,ax=axis();names=['conpath','marginal_shuffle'];values=[analysis['methods'][n]['aggregate']['brier'] for n in names]
    ax.bar([0,1],[v['mean'] for v in values],yerr=[v['sample_sd'] for v in values],color=[TEAL,AMBER],width=.42,capsize=4)
    for i,v in enumerate(values):ax.text(i,v['mean']+v['sample_sd']+.009,f"{v['mean']:.4f}",ha='center',fontsize=15,weight='bold')
    ax.set(xticks=[0,1],xticklabels=['保留空间相关性','逐格打散空间相关性'],ylabel='路径事件 Brier 分数 ↓',ylim=(0,.21))
    save(fig,'dependence','每一格的概率不变，路径判断仍会变','两组保留完全相同的单元格经验概率；只打散跨位置的共同变化。评估干预，未重新训练。','flatlands_clean_paper_analysis.json')
    return chart_data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--charts-only',action='store_true');args=parser.parse_args()
    ASSETS.mkdir(parents=True,exist_ok=True)
    charts=make_charts(); mobile=make_charts(mobile=True)
    for name, chart in charts.items():
        chart['mobile_svg']=mobile[name+'-mobile']['svg']
        chart['mobile_pdf']=mobile[name+'-mobile']['pdf']
    print('Chinese desktop/mobile charts rendered',flush=True)
    if args.charts_only:
        path=DATA/'site_visuals_zh.json'
        if path.exists():
            payload=read(path)
            changed={str(Path(__file__).relative_to(ROOT)), 'site/data/flatlands_clean_paper_analysis.json', 'site/data/flatlands_clean_checkpoint_controls.json'}
            for name,digest in payload['sources'].items():
                if name not in changed:assert sha(ROOT/name)==digest, f'Non-chart source changed: {name}; run the full visual build'
            payload['charts']=charts
            for name in changed:payload['sources'][name]=sha(ROOT/name)
            payload['assets']={f'assets/zh/{p.name}':{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ASSETS.iterdir()) if p.is_file()}
            write(path,payload)
        return
    gallery=make_gallery();print('Training-only galleries and 18-frame sequence rendered',flush=True)
    examples=make_examples(args.device);print('Two frozen validation examples rendered without endpoint connectors',flush=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), SELECTION,QUERIES,MANIFEST,DATA/'flatlands_clean_paper_analysis.json',DATA/'flatlands_clean_checkpoint_controls.json',DATA/'flatlands_k128_clean_candidate.json',ROOT/'src/pathrel/model.py',ROOT/'src/pathrel/flatlands_data.py',ROOT/'src/pathrel/unscenes3d.py']}
    payload={'schema_version':1,'language':'zh-CN','training_started':False,'test_evaluated':False,'sources':sources,'palette':{'free':FREE,'blocked':BLOCKED,'unknown':UNKNOWN,'invalid':'light checker','probability_0':'#f1ecd4','probability_1':TEAL,'start':'blue circle S','goal':'brown diamond G','endpoint_connector':False},'gallery':gallery,'examples':examples,'charts':charts,'method_labels':METHODS,'assets':{f'assets/zh/{p.name}':{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ASSETS.iterdir()) if p.is_file()}}
    write(DATA/'site_visuals_zh.json',payload)
    print(json.dumps({'flatlands_gallery':len(gallery['flatlands']),'unscenes3d_gallery':len(gallery['unscenes3d']),'sequence_frames':len(gallery['sequence']),'examples':len(examples),'assets':len(payload['assets']),'test_evaluated':False}))


if __name__=='__main__':main()
