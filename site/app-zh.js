(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const all = (selector) => [...document.querySelectorAll(selector)];
  const escape = (text) => String(text).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const zoom = (path, caption, eager = false) => `<a class="zoomable" href="${escape(path)}" data-zoom data-caption="${escape(caption)}"><img src="${escape(path)}" alt="${escape(caption)}" loading="${eager ? 'eager' : 'lazy'}" decoding="async"><span class="zoom-hint" aria-hidden="true">点击放大 ↗</span></a>`;
  const choose = (selector, value, attribute) => all(selector).forEach((button) => {
    const selected = button.dataset[attribute] === String(value);
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  const dialog = $('#image-dialog');
  let focusBeforeDialog;
  document.addEventListener('click', (event) => {
    const link = event.target.closest('[data-zoom]');
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (typeof dialog.showModal !== 'function') return;
    event.preventDefault();focusBeforeDialog = link;
    $('#dialog-image').src = link.href;
    $('#dialog-image').alt = link.dataset.caption || link.querySelector('img')?.alt || '放大图片';
    $('#dialog-image').dataset.chart = String(link.id === 'chart-open');
    $('#dialog-caption').textContent = $('#dialog-image').alt;
    $('#dialog-source').href = link.href;
    dialog.showModal();
  });
  $('#dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {if (event.target === dialog) {const r=dialog.getBoundingClientRect();if(event.clientX<r.left || event.clientX>r.right || event.clientY<r.top || event.clientY>r.bottom) dialog.close();}});
  dialog.addEventListener('close', () => focusBeforeDialog?.focus());

  fetch('data/site_visuals_zh.json').then((response) => {if (!response.ok) throw new Error('visual snapshot unavailable');return response.json();}).then((data) => {
    let exampleIndex=0, variant='correlated', dataset='flatlands', galleryIndex=0, frameIndex=0, timer;
    function showExample() {
      const row=data.examples[exampleIndex], model=variant==='correlated'?'ConPath':'独立对照';
      const entries=[['observed','① 机器人已经看到的地图：绿色可通行、深灰阻挡、浅灰未知。'],[variant,`② ${model} 推测：色条表示每个格子的可通行概率。`],[variant+'_sample','③ 固定的第一次随机补全：这是一个完整世界。'],['reference','④ 完整参考地图：用于检查实际是否存在通路。']];
      $('#example-panels').innerHTML=entries.map(([key,label])=>zoom(row.panels[key],label,true)).join('');
      $('#example-panels').scrollLeft=0;choose('[data-step]',0,'step');
      $('#example-verdict').textContent=`参考地图：${row.target?'存在通路':'不存在通路'}`;
      $('#example-probability').textContent=`${model} 预测通路概率 ${(row.event_probability[variant]*100).toFixed(1)}%`;
      $('#example-explanation').textContent=row.target?(variant==='correlated'?'ConPath 在这个示例中给出了较高的通路概率。单个格子看起来可走还不够，整张补全地图中的空间结构也会影响是否连通。':'独立对照在这个示例中低估了通路概率。虽然均值概率地图看起来接近，独立采样可能破坏整条通路的共同结构。'):'这是一个需要保留的失败例子：参考地图判定无路，但两个模型仍给出较高的通路概率。ConPath 的概率较低，也不意味着它已经判断正确。';
      $('#example-source').textContent=`FlatLands / ${row.source} · ${row.global_id} · 机器人半径 ${row.radius_cells} 格（按发布包元数据换算为 ${row.radius_m.toFixed(2)} 米，物理尺度待核对）· 模型种子 ${row.seed}。固定采样示例与原验证统计使用不同随机流；通路概率来自原验证结果。按标签挑选的解释性示例，不能代替整体统计。`;
      choose('[data-example]',exampleIndex,'example');
    }
    all('[data-example]').forEach((button)=>button.addEventListener('click',()=>{exampleIndex=Number(button.dataset.example);showExample();}));
    $('#model-select').addEventListener('change',(event)=>{variant=event.target.value;showExample();});
    all('[data-step]').forEach((button)=>button.addEventListener('click',()=>{const container=$('#example-panels'),panel=container.children[Number(button.dataset.step)];container.scrollTo({left:panel.offsetLeft-container.children[0].offsetLeft,behavior:'smooth'});choose('[data-step]',button.dataset.step,'step');}));
    $('#example-panels').addEventListener('scroll',()=>{const container=$('#example-panels');const width=container.children[0]?.getBoundingClientRect().width||1;choose('[data-step]',Math.round(container.scrollLeft/(width+12)),'step');},{passive:true});
    function galleryThumbnails() {
      $('#gallery-thumbnails').innerHTML=data.gallery[dataset].map((row,index)=>`<button class="thumbnail ${index===galleryIndex?'selected':''}" type="button" data-gallery-index="${index}" aria-pressed="${index===galleryIndex}" aria-label="查看场景 ${index+1}：${escape(row.source||row.scene)}"><img src="${escape(row.camera||row.observed)}" alt="" role="presentation" loading="lazy"><span>${escape(row.source||row.scene)}</span></button>`).join('');
    }
    function showGallery() {
      const row=data.gallery[dataset][galleryIndex], outdoor=dataset==='unscenes3d';
      $('#gallery-stage').classList.toggle('outdoor',outdoor);
      $('#gallery-title').textContent=outdoor?`UnScenes3D · ${row.scene}`:`FlatLands · ${row.source}`;
      $('#gallery-count').textContent=`${String(galleryIndex+1).padStart(2,'0')} / ${String(data.gallery[dataset].length).padStart(2,'0')}`;
      $('#gallery-description').textContent=outdoor?'照片与两张俯视地图来自同一时刻。地图是激光雷达观测与占用标注的投影；棋盘格区域不参与可通行判断。':'左边是模型能够看到的部分，右边是用于核对的完整参考地图。这些是俯视栅格图，不是相机照片。';
      const camera=outdoor?`<figure><div class="camera-stage">${zoom(row.camera,`${row.scene} 的原始相机照片；时间 ${row.timestamp}`,true)}</div><figcaption>相机原始画面<span>帮助理解场景，未直接输入当前模型</span></figcaption></figure>`:'';
      $('#gallery-stage').innerHTML=camera+['observed','reference'].map((key)=>zoom(row[key],`${outdoor?'UnScenes3D':'FlatLands'} / ${row.scene}：${key==='observed'?'已观测地图':'完整参考地图'}`,true)).join('');
      $('#gallery-source').textContent=outdoor?`训练集 · ${row.location} / ${row.scene} · 时间戳 ${row.timestamp} · 每格 0.30 米。原始相机图片保持数据集发布字节。`:`训练集 · ${row.source} / ${row.scene} · ${row.id} · 每格 ${row.resolution_m.toFixed(2)} 米。模型输入与完整参考严格区分。`;
      choose('[data-dataset]',dataset,'dataset');galleryThumbnails();
    }
    all('[data-dataset]').forEach((button)=>button.addEventListener('click',()=>{dataset=button.dataset.dataset;galleryIndex=0;showGallery();}));
    $('#gallery-thumbnails').addEventListener('click',(event)=>{const button=event.target.closest('[data-gallery-index]');if(button){galleryIndex=Number(button.dataset.galleryIndex);showGallery();}});
    const sequence=data.gallery.sequence;
    function showFrame() {
      const row=sequence[frameIndex];
      $('#sequence-stage').innerHTML=`<figure><div class="camera-stage">${zoom(row.camera,`原始相机照片 · ${row.timestamp}`,true)}</div><figcaption>① 相机原始画面<span>${escape(row.scene)} · ${escape(row.timestamp)}</span></figcaption></figure>`+[['observed','② 激光雷达观测'],['reference','③ 数据集参考地图']].map(([key,label])=>`<figure>${zoom(row[key],`${label} · ${row.timestamp}`,true)}<figcaption>${label}</figcaption></figure>`).join('');
      $('#sequence-frame').value=frameIndex;$('#sequence-counter').textContent=`${String(frameIndex+1).padStart(2,'0')} / ${sequence.length}`;
      const next=sequence[(frameIndex+1)%sequence.length];[next.camera,next.observed,next.reference].forEach((src)=>{const image=new Image();image.src=src;});
    }
    function stopPlayback(){clearInterval(timer);timer=undefined;$('#sequence-play').textContent='播放浏览';$('#sequence-play').setAttribute('aria-pressed','false');}
    $('#sequence-play').addEventListener('click',()=>{if(timer){stopPlayback();return;}if(frameIndex===sequence.length-1)frameIndex=0;showFrame();$('#sequence-play').textContent='暂停浏览';$('#sequence-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{if(frameIndex>=sequence.length-1){stopPlayback();return;}frameIndex++;showFrame();},1000);});
    $('#sequence-frame').addEventListener('input',(event)=>{stopPlayback();frameIndex=Number(event.target.value);showFrame();});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();});
    new IntersectionObserver((entries)=>{if(!entries[0].isIntersecting)stopPlayback();},{threshold:0}).observe($('#sequence-stage'));
    const explanations={brier:'Brier 衡量预测概率与实际结果的偏差，0 最好。横线表示训练种子标准差，不是置信区间。所有对照使用相同验证查询。',risk:'横轴是模型愿意接受多少查询，纵轴是接受后却实际无路的比例，越低越好。应在相同覆盖率下比较。浅色带为训练种子标准差；30% 覆盖率的配对区间未证实稳定的风险改善。',reliability:'横轴是模型说“有路”的概率，纵轴是这些查询实际有路的比例。越接近灰色对角线越可信；每个点是一个概率分箱的加权统计，不是单次预测。',sampling:'横轴是评估时的地图采样数 K，纵轴是路径概率误差。这里复用已训练模型；增加采样不等于重新训练。误差线是三个训练种子的标准差。',dependence:'两组的每格经验概率完全相同。仅打散不同位置之间的共同变化，路径误差就增大了；这是一项评估时干预，不是新的训练消融。'};
    all('button[data-chart]').forEach((button)=>button.addEventListener('click',()=>{const name=button.dataset.chart,chart=data.charts[name];$('#chart-image').src=chart.svg;$('#chart-mobile').srcset=chart.mobile_svg;$('#chart-image').alt=`${chart.title}。${chart.note}`;$('#chart-open').href=matchMedia('(max-width: 600px)').matches?chart.mobile_svg:chart.svg;$('#chart-open').dataset.caption=chart.title;$('#chart-pdf').href=chart.pdf;$('#chart-caption').textContent='怎么看：'+explanations[name];choose('button[data-chart]',name,'chart');}));
    showExample();showGallery();document.documentElement.dataset.interactiveReady='true';
  }).catch((error)=>{console.error(error);$('#interaction-error').hidden=false;});
})();
