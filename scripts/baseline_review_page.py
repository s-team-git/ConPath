"""Current baseline findings, separate from historical charts and paper references."""
import hashlib
import html
import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / 'site'


def baseline_review():
    data = json.loads((SITE / 'data/baseline_review_zh.json').read_text())
    analysis = json.loads((SITE / 'data/current_baseline_k4_analysis.json').read_text())
    refs = json.loads((SITE / 'data/published_baseline_reference.json').read_text())
    assert not data['external_superiority_established'] and not data['old_parent_isolation_passed']
    assert len(refs['rows']) == 90 and not refs['direct_conpath_ranking_allowed']
    for asset in data['assets']:
        assert hashlib.sha256((SITE / asset['path']).read_bytes()).hexdigest() == asset['sha256']
    rows = []
    for key, name in data['labels'].items():
        m = analysis['methods'][key]
        def fmt(metric, percent=False):
            a = m[metric]; scale = 100 if percent else 1
            value = f'{a["mean"]*scale:.2f}%' if percent else f'{a["mean"]:.5f}'
            sd = f'{a["seed_sd"]*scale:.2f}%' if percent and a['seed_sd'] is not None else f'{a["seed_sd"]:.5f}' if a['seed_sd'] is not None else None
            return value + (' ± '+sd if sd is not None else '')
        rows.append(f'<tr data-current-baseline="{key}"><th scope="row">{html.escape(name)}</th><td>{m["samples"]}</td><td>{fmt("brier")}</td><td>{fmt("risk30", True)}</td><td>{fmt("mean_iou")}</td></tr>')
    paper_rows = []
    for name, label in [('LaMa-Ens.', 'LaMa集成'), ('Diffusion', '条件扩散'), ('Flow Match.', '条件流匹配'), ('FM+XAttn', '流匹配＋交叉注意力')]:
        cells = []
        for metric in ['MES', 'mean_of_K_IoU']:
            r = next(r for r in refs['rows'] if r['table'] == '4' and r['split'] == 'ID' and r['method'] == name and r['metric'] == metric)
            cells.append(f'<td>{r["mean"]:.3f} ± {r["reported_std"]:.3f}</td>')
        paper_rows.append(f'<tr data-published-reference="{name}"><th scope="row">{label}</th>'+''.join(cells)+'</tr>')
    return f'''<div id="baseline-review" class="baseline-review">
      <p class="eyebrow">最新评估 · 2026.09.09</p><h3>还不能说超过论文；先把对比和数据划分做扎实。</h3>
      <p>已复用现有检查点完成四次采样评估，加入三种无需训练的规则，并整理90条论文公开成绩。本轮没有新增长训练。</p>
      <div class="audit-notice"><strong>数据隔离更正：27 / 160条旧验证观测与训练共享上一级地点。</strong><p>同一建筑的不同房间、全景或重复扫描不能当作完全独立环境。旧数值仍可用于诊断，但不能证明对新建筑的泛化；旧的子场景统计区间也有相同限制。</p><p>读取范围也需更正：旧训练／验证包含原目录test中的8／5条观测，本轮回放读到了其中5条；历史查询审计还检查过32条ScanNet++观测。撤回“物理test从未读取”的说法，最终测试须重新审核未触碰资格。<a href="data/flatlands_read_scope_erratum.json">读取范围更正记录 ↗</a></p></div>
      <figure class="main-chart baseline-chart"><a href="assets/zh/baseline-review/current-controls.svg" data-zoom data-caption="当前基线诊断：柱越短越好，横线为训练标准差；旧划分存在建筑重叠。"><picture><source media="(max-width: 600px)" srcset="assets/zh/baseline-review/current-controls-mobile.svg"><img src="assets/zh/baseline-review/current-controls.svg" alt="ConPath四次采样Brier为0.0801，全可通行规则为0.0724；相同30%覆盖率误判为4.89%和13.43%。" loading="lazy"></picture></a><figcaption>每行写出方法名称，颜色对应同一方法。横线表示三次训练的标准差，不是建筑置信区间。全阻挡规则及完整数字见下表。<a href="assets/zh/baseline-review/current-controls.pdf">下载PDF ↗</a></figcaption></figure>
      <p class="reading-note"><strong>简单规则的Brier点估计更低，但误判风险更高。</strong>ConPath并非全面领先：地图IoU低于确定性网络，30%覆盖率风险与独立模型接近。点估计不等于显著胜负；这些比较还受旧划分限制。</p>
      <details id="current-baseline-details" class="plain-details"><summary>展开本地六种方法的统一评估</summary><div class="detail-body"><div class="table-scroll"><table class="current-baseline-table"><caption>旧队列诊断 · 160张地图／142个事件子场景／4,224个查询；±为三次训练标准差</caption><thead><tr><th>方法</th><th>输出数</th><th>Brier ↓</th><th>30%覆盖率误判 ↓</th><th>平均地图IoU ↑</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>确定性方法只输出一张，不复制成四张；规则没有训练标准差。K=4取原随机流的前四张，没有按真实答案挑选。旧K=128表保留在下方，二者采样预算不同。</p><p><a href="data/current_baseline_k4_analysis.json">完整本地统计 ↗</a> · <a href="data/baseline_review_zh.json">回放检查与统计适用范围 ↗</a></p></div></details>
      <details id="paper-reference-details" class="plain-details"><summary>展开论文公开成绩：参考表，不是与本项目的排名</summary><div class="detail-body"><p>FlatLands v3 表4、域内测试、K=4。MES评价整组地图，不是通路Brier；论文的原始训练/测试划分与本地队列不同，因此不能直接比较优劣。</p><div class="table-scroll"><table class="published-reference-table"><caption>仅引用论文原表；±保持原文含义，不能当作本地训练标准差</caption><thead><tr><th>论文方法</th><th>地图MES ↓</th><th>平均地图IoU ↑</th></tr></thead><tbody>{''.join(paper_rows)}</tbody></table></div><p>按最新要求，优先引用可比较的公开成绩，其次复用作者预测或权重，再做必要的同协议适配训练。自定义通路指标目前没有对应论文分数，不能从地图均值换算。</p><p><a href="https://arxiv.org/html/2603.16016v3">核对原论文 ↗</a> · <a href="data/published_baseline_reference.csv">90条公开参考CSV ↗</a></p></div></details>
      <details id="data-repair-details" class="plain-details"><summary>展开分组修正、历史关键帧与Transformer分析</summary><div class="detail-body"><p>重叠观测：Matterport3D 19条、3RScan 5条、ZInD 2条、ScanNet 1条、ARKitScenes 0条。已生成215,289条观测的父级分组候选，训练／校准／开发验证的地点交叉为零，53条缺失地点编号的观测单独隔离。原物理域内测试也存在地点交叉，最终测试方案仍需核定。</p><p>室外当前硬观测造成的Brier下界已达0.4757，实际误差0.5114；先修正观测置信度、地面高度与可通行定义，再增加上下文。训练集27对历史帧中，候选对齐提高平均点云重合率，但只在11对改善中位距离，尚不能认为位姿质量全面通过。</p><p>后续采用四组：单帧卷积、单帧注意力、历史几何融合、历史注意力融合；同样帧数与数据预算下比较。另区分增加独立地点和增加同地点观测的效果。没有使用未来帧，也没有声称Transformer已经提升成绩。</p><p><a href="data/flatlands_parent_group_audit.json">父级分组审计 ↗</a> · <a href="https://github.com/s-team-git/ConPath/blob/main/BASELINE_REVIEW_ZH.md">完整中文分析与下一步实验 ↗</a></p></div></details>
      </div>'''
