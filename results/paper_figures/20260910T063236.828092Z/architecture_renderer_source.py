#!/usr/bin/env python3
"""Correct only the implemented conditioning/sampling order; preserve other figures."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paper_base', OUT/'build_saved_figures.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def architecture(folder, evidence):
    plt = base.plt
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, title, body, color='#edf5f3'):
        ax.add_patch(base.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#7e9691',lw=1.2))
        ax.text(x+w/2,y+h-.04,title,ha='center',va='top',fontsize=13,fontweight='bold')
        ax.text(x+w/2,y+h-.087,body,ha='center',va='top',fontsize=10,linespacing=1.45)
    def arrow(a, b, dashed=False):
        ax.add_patch(base.FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,color='#436c65',lw=1.5,linestyle='--' if dashed else '-'))
    box(.025,.70,.21,.23,'共同观测与支持','3通道：可通行 / 阻挡 / 未知\n有效支持mask单独约束\n不输入完整参考地图')
    box(.285,.70,.25,.23,'TinyBEVUNet 编码器','内部添加二维坐标平面\n16 → 32 → 64通道，两次下采样\n双线性上采样 + 跳连\n池化全局上下文相加')
    box(.585,.70,.39,.23,'相关 Gaussian logit','μ：两类logit位置，不是边际概率\nB：4维全局Gaussian因子；σ：局部尺度\nL_k = μ + B z_k / √4 + σ · filter(ε_k)\n局部核5×5；边缘按卷积权重归一方差')
    box(.585,.38,.39,.24,'采样前：已知证据 / 支持条件约束','已观测类别的logit设为+10，另一类设为-10\nPathRelNet将支持范围外列入已知障碍\n约束先于类别抽样；不是采样后重写硬地图\n条件均值图的独立投影属于评价器诊断')
    box(.285,.38,.25,.24,'原版类别采样','逐单元独立Gumbel；噪声尺度a=1\nU截断：[1e-6, 1-1e-6]\n硬前向：argmax(L_k + G_k)\nstraight-through反向温度0.7\n注册设置下已审计硬样本零违规')
    box(.025,.38,.21,.24,'K张二值世界','W_1, W_2, ..., W_K\n跨世界使用独立随机流\n相关来自Gaussian因子/局部logit\n逐格mean(W_k)是地图频率\n不是路径事件分数')
    box(.025,.075,.21,.20,'精确任务几何','固定S / G / 半径r（格）\n圆盘足迹 + 四邻域\n超出地图视为障碍\n每世界事件ρ_k ∈ {0,1}')
    box(.285,.075,.25,.20,'事件频率','p(S,G,r) = mean_k ρ_k\n不是预测一条路线\n单个世界不等于真实地图\n验证用完整精确连通性')
    box(.585,.075,.39,.20,'训练监督（只用于训练）','隐藏图BCE + 0.1 variogram + 2 × 事件U统计\n事件硬前向由完整精确oracle修正\n反向保留最多256步松弛传播\n这是代理梯度，不是精确硬连通性梯度','#f7f0e6')
    for a,b in [((.236,.81),(.285,.81)),((.535,.81),(.585,.81)),((.78,.70),(.78,.62)),((.585,.5),(.535,.5)),((.285,.5),(.236,.5)),((.13,.38),(.13,.275)),((.236,.175),(.285,.175))]: arrow(a,b)
    arrow((.585,.175),(.535,.175),True)
    fig.suptitle('最终ConPath代码路径：相关logit → 条件约束 → 独立类别抽样 → 足迹事件',fontsize=17,y=.995)
    base.footnote(fig,'非测量架构图；仅对应parent-pilot最终原版方法，scene/query/seed具体数值不适用。\n已知类别间隔20与a=1截断Gumbel共同保留硬证据；不声称任意噪声尺度均成立。\ncoherent为另训的补充候选，不在此主路径；no-global保留编码器全局上下文，no-event只关闭事件训练项。',evidence)
    base.save(fig,folder,'01_architecture','最终原版ConPath的实际代码顺序：先按已知类别/支持范围设logit±10，再加入尺度1的截断独立Gumbel并采样；不存在原硬地图采样后的通用投影步骤。',
        'Implemented original ConPath pipeline. Observed and support-invalid cells are conditioned through logits of +10/-10 before unit-scale clipped independent Gumbel sampling. Correlation arises from Gaussian global factors and local logit fields. Exact hard-event correction uses a bounded relaxed backward surrogate; coherent categorical noise is a separately trained supplementary variant.',
        evidence=evidence,scope='implementation schematic; no experiment score')


def main():
    base.plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
                             'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
    previous=ROOT/json.loads((OUT/'current.json').read_text())['snapshot']
    manifest=json.loads((previous/'manifest.json').read_text())
    source_hashes=dict(manifest['source_hashes'])
    source_hashes[str(Path(__file__).relative_to(ROOT))]=base.digest(Path(__file__))
    evidence=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    folder=OUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');folder.mkdir(exist_ok=False)
    reused=[]; figures=[]
    for figure in manifest['figures']:
        if figure['name']=='01_architecture':continue
        for artifact in figure['files']:
            src=ROOT/artifact['path'];dst=folder/src.name
            if base.digest(src)!=artifact['sha256']:raise ValueError('Previous immutable figure changed')
            shutil.copy2(src,dst);artifact['path']=str(dst.relative_to(ROOT))
            if base.digest(dst)!=artifact['sha256']:raise ValueError('Copy changed figure pixels/data')
            reused.append({'old':str(src.relative_to(ROOT)),'new':str(dst.relative_to(ROOT)),'sha256':artifact['sha256']})
        figures.append(figure)
    architecture(folder,evidence)
    for name in ['metrics_from_saved_predictions.json','selection_before_rendering.json']:
        shutil.copy2(previous/name,folder/name)
    shutil.copy2(previous/'renderer_source.py',folder/'base_renderer_source.py')
    shutil.copy2(Path(__file__),folder/'architecture_renderer_source.py')
    manifest['figures']=[base.INVENTORY[-1]]+figures
    manifest['created_utc']=datetime.now(timezone.utc).isoformat()
    manifest['source_bundles']={manifest['source_bundle_sha256']:manifest['source_hashes'],evidence:source_hashes}
    manifest['source_hashes']=source_hashes
    manifest['architecture_source_bundle_sha256']=evidence
    manifest['revision']={'reason':'Correct original implementation conditioning-before-sampling order only',
                          'previous_snapshot':str(previous.relative_to(ROOT)),'previous_manifest_sha256':base.digest(previous/'manifest.json'),
                          'unchanged_other_figure_artifacts':reused,'unchanged_other_figure_artifact_count':len(reused),
                          'new_inference':False,'new_model_evaluation':False,
                          'base_renderer_source_sha256':base.digest(folder/'base_renderer_source.py'),
                          'architecture_renderer_source_sha256':base.digest(folder/'architecture_renderer_source.py')}
    base.write_json(folder/'manifest.json',manifest)
    base.write_json(OUT/'current.json',{'snapshot':str(folder.relative_to(ROOT)),'manifest_sha256':base.digest(folder/'manifest.json')})
    doc=ROOT/'PAPER_FIGURES.md';text=doc.read_text().replace(str(previous.relative_to(ROOT)),str(folder.relative_to(ROOT))).replace(previous.name,folder.name)
    text=text.replace('按实际模型、解码器和父级地点训练脚本绘制；硬前向与松弛反向明确区分。',base.INVENTORY[-1]['caption_zh'])
    old_caption=next(f['caption_en'] for f in json.loads((previous/'manifest.json').read_text())['figures'] if f['name']=='01_architecture')
    text=text.replace(old_caption,base.INVENTORY[-1]['caption_en'])
    doc.write_text(text)
    print(json.dumps({'snapshot':str(folder.relative_to(ROOT)),'figures':len(manifest['figures']),'copied_artifacts_identical':len(reused),
                      'manifest_sha256':base.digest(folder/'manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
