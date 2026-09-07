# ConPath 中文研究主页

主页按“方法读图 → 数据示例 → 当前结果 → 数据集与原论文对照”组织，以中文解释为主。
参考用户提供的 [Lightweight-3DGS](https://s-team-git.github.io/Lightweight-3DGS/) 单栏论文主页，
重新实现静态 HTML/CSS/JavaScript；无需前端框架、在线字体或第三方脚本。

## 图片与交互

- 两个固定验证示例，可切换相关模型与独立对照；单元格概率和整体通路概率明确区分。
- 蓝色圆点 S 是起点，棕色菱形 G 是目标；不把查询端点画成直连规划路线。
- 所有地图附中文颜色图例，概率地图自带 0–1 色条。手机使用四步大图切换；图片可放大。
- 6 个 FlatLands 训练样本覆盖本地 5 个来源；6 个 UnScenes3D 训练场景覆盖 3 个地点。
- 18 个同场景采样帧同步显示相机、观测地图、参考地图，支持播放、暂停和拖动。
  这是每秒一帧的数据浏览，不是实时推理或原始录像速度；相机照片保持原始发布字节。
- 五组中文图表分别输出桌面版、手机版 SVG/PDF，复用已审计的数值，不改变指标。
  详细九方法表默认折叠，失败和无优势的结果明确保留。

`index.html` 使用 `styles-zh.css` 与 `app-zh.js`。以前的页面在 `archive/index.html`，
包括旧英文图表、审计过程和带查询连线的 TUM 几何演示。它们不再出现在当前主页。

## 重建与检查

```bash
# 只回放已完成的模型与数据，不训练；默认 --device cpu，也可用 cuda 加速绘图。
PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/build_site_visuals.py --device cuda
/home/hairo/miniconda3/bin/python3.13 scripts/build_site_page.py
PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/audit_site_visuals.py
PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/audit_paper_evidence.py --output results/site_redesign_20260907/paper_evidence_audit.json
```

最新绘图与逐图来源：`data/site_visuals_zh.json`。它保存源文件、检查点、原始相机图、
统计快照和每个导出文件的哈希。原始数据与模型权重仍在忽略目录，没有加入 Git。
相机照片共 23 张唯一原始帧：6 个图库场景和 18 帧时序中有 1 张重合。
结果图的三次训练误差线为种子标准差，不是置信区间。

如果重新运行旧的 `build_paper_evidence.py` 更新统计表，应随后运行
`build_site_page.py`，恢复当前中文模板和指标单位。

本地预览：`python3 -m http.server 8765 --bind 127.0.0.1 --directory site`。
启动带 `--remote-debugging-port=9223` 的 Chrome 后，运行
`node scripts/check_site_browser.mjs` 检查 1440/390 像素下的图片、图表、图库、播放、放大、
手机步骤切换、键盘关闭与页面溢出。记录在 `results/site_redesign_20260907/`。

## 发布与当前研究状态

GitHub Pages 通过 `.github/workflows/deploy-pages.yml` 发布，每次推送 `main` 触发部署。
主实验是 FlatLands 场景隔离的非官方验证划分，UnScenes3D 仍为两场景诊断；正式测试未评估。
新增训练消融保持用户请求的暂停状态，绘图命令不恢复该训练。

数据集选择、原方法实际使用的数据及媒体署名见
[DATASET_CHOICE_ZH.md](../DATASET_CHOICE_ZH.md)。FlatLands 的上游原始 RGB-D 等源素材未被重新发布，
新图库显示发布包中的派生地图。UnScenes3D 相机图片来自数据集发布文件，未改编其论文插图。
