(() => {
  'use strict';
  const oldSections = ['baseline-review', 'training-ablations', 'ablation-details', 'ablation-examples', 'datasets', 'comparison', 'external-progress', 'flatlands-external-progress'];
  const route = () => {
    if (oldSections.includes(location.hash.slice(1))) {
      location.replace('research.html' + location.search + location.hash);
      return true;
    }
    return false;
  };
  if (route()) return;
  window.addEventListener('hashchange', route);
  const $ = selector => document.querySelector(selector);
  const all = selector => [...document.querySelectorAll(selector)];
  const choose = (selector, key, value) => all(selector).forEach(button => {
    const selected = button.dataset[key] === String(value);
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  try {
    const data = JSON.parse($('#home-data').textContent);
    let index = 0, model = 'correlated', view = 'sample';
    const gallery = $('#output-gallery');
    const dialog = $('#model-dialog');
    let previousFocus;
    function render() {
      const row = data.examples[index];
      const modelName = model === 'correlated' ? 'ConPath' : '独立单元对照';
      const predictionKey = model + (view === 'sample' ? '_sample' : '');
      const panels = [
        ['input', row.panels.observed, '① 模型输入：已观测地图，浅灰区域未知。'],
        ['prediction', row.panels[predictionKey], `② ${modelName}预测：${view === 'sample' ? '一次实际采样的完整地图。' : '每格的可通行概率，米白到青绿表示0到1。'}`],
        ['reference', row.panels.reference, '③ 真实参考：完整地图，只用于核对结果。'],
      ];
      for (const [key, source, caption] of panels) {
        const image = $('#image-' + key), link = image.closest('a');
        image.src = source; image.alt = caption; link.href = source; link.dataset.caption = caption;
      }
      $('[data-panel="prediction"] figcaption p').textContent = view === 'sample' ? '一次实际采样的完整地图。' : '每格可通行的概率，米白0 → 青绿1。';
      $('#view-explanation').textContent = view === 'sample' ? '绿色表示可通行，深灰表示阻挡。' : '颜色表示单元格概率，不是整条通路的概率。';
      $('#case-probability').textContent = (row.event_probability[model] * 100).toFixed(1) + '%';
      $('#case-reading').classList.toggle('failure', !row.target);
      const outcome = row.target ? (model === 'correlated' ? '参考地图有路；ConPath给出了较高的通路概率。' : '参考地图有路，但独立模型低估了通路概率。') : '这是一个失败例子：参考地图无路，但模型仍给出较高的通路概率。';
      $('#case-explanation').textContent = outcome + (view === 'sample' ? `本张实际补全${row.displayed_world_event[model] ? '存在' : '不存在'}通路。` : '概率图展示每格的信心，不能单凭颜色判断是否连通。');
      $('#case-source').textContent = `示例 ${row.global_id} · ${row.source} · 模型种子${row.seed} · 机器人半径${row.radius_cells}格。通路概率来自原128次采样评估，示例图使用另一组固定随机流；均为同一检查点。`;
      choose('[data-case]', 'case', index); choose('[data-view]', 'view', view);
    }
    all('[data-case]').forEach(button => button.addEventListener('click', () => { index = Number(button.dataset.case); render(); }));
    all('[data-view]').forEach(button => button.addEventListener('click', () => { view = button.dataset.view; render(); }));
    $('#home-model').addEventListener('change', event => { model = event.target.value; render(); });
    all('[data-jump]').forEach(button => button.addEventListener('click', () => {
      const i = Number(button.dataset.jump);
      gallery.scrollTo({left:gallery.children[i].offsetLeft-gallery.children[0].offsetLeft,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
      choose('[data-jump]', 'jump', i);
    }));
    gallery.addEventListener('scroll', () => {
      const width = gallery.children[0].getBoundingClientRect().width;
      choose('[data-jump]', 'jump', Math.round(gallery.scrollLeft/(width+12)));
    }, {passive:true});
    document.addEventListener('click', event => {
      const link = event.target.closest('[data-image]');
      if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || typeof dialog.showModal !== 'function') return;
      event.preventDefault(); previousFocus = link;
      $('#model-dialog-image').src = link.href;
      $('#model-dialog-image').alt = link.dataset.caption;
      $('#model-dialog-caption').textContent = link.dataset.caption;
      $('#model-dialog-source').href = link.href; dialog.showModal();
    });
    $('#model-dialog-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => previousFocus?.focus());
    render(); document.documentElement.dataset.homeReady = 'true';
  } catch (error) {
    console.error(error); $('#home-error').hidden = false;
  }
})();
