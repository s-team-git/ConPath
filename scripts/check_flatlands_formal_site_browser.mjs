// Read-only real-browser audit of the formal diagnostic page. No model inference.
// Start a dedicated local HTTP server and headless Chrome with GPU disabled first.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';

const base = process.argv[2] || 'http://127.0.0.1:8777/formal.html';
const output = process.argv[3] || 'results/flatlands_formal_site_publication_v1';
const debug = process.argv[4] || 'http://127.0.0.1:9237';
const homeOnly = process.argv.includes('--home-only');
const check = (ok, message) => {if (!ok) throw new Error(message);};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const tab = await fetch(debug + '/json/new?about:blank', {method: 'PUT'}).then(r => r.json());
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
let nextId = 0;
const pending = new Map(), exceptions = [];
ws.onmessage = ({data}) => {
  const message = JSON.parse(data);
  if (message.id) {
    const item = pending.get(message.id); pending.delete(message.id);
    message.error ? item?.reject(message.error) : item?.resolve(message.result);
  } else if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params);
};
const command = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++nextId; pending.set(id, {resolve, reject}); ws.send(JSON.stringify({id, method, params}));
});
const evaluate = async expression => {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  check(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
  return result.result.value;
};
async function screenshot(name) {
  const result = await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  const bytes = Buffer.from(result.data, 'base64');
  await fs.writeFile(path.join(output, name + '.png'), bytes, {flag: 'wx'});
  return {path: name + '.png', sha256: sha(bytes)};
}

const receipt = {passed: false, scope: 'Diagnostic website rendering and published values; no scientific eligibility',
  created_utc: new Date().toISOString(), base, gpu_model_inference_performed: false, new_test_images_opened: 0,
  viewports: [], screenshots: []};
try {
  await command('Page.enable'); await command('Runtime.enable'); await command('Network.enable');
  await command('Network.setCacheDisabled', {cacheDisabled: true});
  for (const width of [1440, 390]) {
    await command('Emulation.setDeviceMetricsOverride', {width, height: width === 1440 ? 1080 : 844, deviceScaleFactor: 1, mobile: width < 700});
    await command('Page.navigate', {url: homeOnly ? new URL('index.html', base).href : base});
    let ready = false;
    for (let attempt = 0; attempt < 100; attempt++) {
      ready = await evaluate(homeOnly
        ? 'document.readyState === "complete" && Boolean(document.querySelector("#next"))'
        : 'document.readyState === "complete" && Boolean(document.querySelector("#gallery"))');
      if (ready) break;
      await delay(100);
    }
    check(ready, 'Formal page did not finish loading');
    if (homeOnly) {
      const state = await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
        lang:document.documentElement.lang,next:document.querySelector('#next').innerText,
        links:[...document.querySelectorAll('a[href="formal.html"]')].map(a=>({label:a.innerText,url:a.href}))})`);
      check(state.lang === 'zh-CN' && state.scrollWidth <= width + 1, 'Home language or horizontal overflow');
      check(state.links.length === 3, 'Missing current formal entry');
      for (const term of ['985', '97', '191', '1000', '5000', '测试'])
        check(state.next.includes(term), 'Current frozen workflow absent: ' + term);
      check(!state.next.includes('当前不追加训练') && !state.next.includes('历史融合'), 'Obsolete current task text');
      receipt.screenshots.push(await screenshot(width + '-home-overview'));
      await evaluate('document.querySelector("#next").scrollIntoView({block:"start",behavior:"instant"})');
      receipt.screenshots.push(await screenshot(width + '-home-next'));
      const linked = await fetch(state.links[0].url); check(linked.ok, 'Formal entry target unavailable');
      const html = await linked.text(); check(html.includes('validation-only'), 'Linked page diagnostic label missing');
      receipt.viewports.push({...state, formal_entry_available:true, formal_html_sha256:sha(html)});
      continue;
    }
    receipt.screenshots.push(await screenshot(width + '-overview'));
    const state = await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
      lang:document.documentElement.lang,rows:[...document.querySelectorAll('#results tbody tr')].map(row=>({
        method:row.querySelector('th .method-en').textContent,cells:[...row.querySelectorAll('td')].map(c=>c.childNodes[0]?.textContent.trim())})),
      figures:document.querySelectorAll('#gallery details').length,
      imageURLs:[...document.querySelectorAll('#gallery img')].map(image=>image.src),
      text:document.body.innerText,jsonURL:document.querySelector('a[href$="/diagnostics.json"]').href})`);
    check(state.lang === 'zh-CN' && state.scrollWidth <= width + 1, 'Language or horizontal page overflow');
    check(!state.text.includes('ours wins') && !state.text.includes('SOTA'), 'Unapproved superiority wording');
    check(state.text.includes('validation-only') && state.text.includes('没有授予论文主表'), 'Diagnostic qualification missing');
    check(!state.imageURLs.some(url => url.includes('11_training_losses')), 'Training image published as validation');
    const response = await fetch(state.jsonURL); check(response.ok, 'Diagnostic JSON unavailable');
    const bytes = Buffer.from(await response.arrayBuffer()), data = JSON.parse(bytes);
    check(data.validation_only === true && data.paper_main_or_superiority_authorized === false,
      'Website diagnostic scope fields differ from the publication schema');
    check(data.final_test_locked === true && data.location_6_locked === true && data.new_physical_test_images_opened === 0,
      'Website test-lock disclosure missing');
    const metrics = ['map_brier','map_nll','event_brier','event_nll','event_ece','false_safe_at_0_8','coverage_at_0_8'];
    check(state.rows.length === Object.keys(data.methods).length && state.figures === data.figures.length, 'Missing method or fixed figure');
    for (const [index, [method, row]] of Object.entries(data.methods).entries()) {
      const actual = state.rows[index]; check(actual.method === row.label, 'Method order or label changed');
      for (const [column, metric] of metrics.entries()) {
        const stats = row.metrics[metric];
        const expected = method === 'direct_query' && metric.startsWith('map_') ? '不适用' : stats.mean === null ? '—'
          : stats.mean.toFixed(4) + (stats.sd !== null ? ' ± ' + stats.sd.toFixed(4) : '');
        check(actual.cells[column + 1] === expected, 'Browser values disagree with audited source: ' + method + '/' + metric);
      }
    }
    await evaluate(`(async()=>{document.querySelectorAll('#gallery details').forEach(item=>item.open=true);
      await Promise.all([...document.querySelectorAll('#gallery img')].map(async image=>{image.loading='eager';await image.decode();}));})()`);
    const gallery = await evaluate(`({scrollWidth:document.documentElement.scrollWidth,
      broken:[...document.querySelectorAll('#gallery img')].filter(image=>!image.complete||!image.naturalWidth).map(image=>image.src),
      imageAltComplete:[...document.querySelectorAll('#gallery img')].every(image=>image.alt.includes('validation-only')&&image.alt.includes('seed=')),
      pdfLinks:document.querySelectorAll('#gallery a[href$=".pdf"]').length})`);
    check(!gallery.broken.length && gallery.scrollWidth <= width + 1 && gallery.imageAltComplete, 'Gallery load, labels or viewport failed');
    check(gallery.pdfLinks === data.figures.length, 'Figure PDF download missing');
    await evaluate('document.querySelector("#gallery details").scrollIntoView({block:"start",behavior:"instant"})');
    receipt.screenshots.push(await screenshot(width + '-first-fixed-case'));
    receipt.viewports.push({width, scrollWidth:state.scrollWidth, methods:state.rows.length, figures:state.figures,
      all_fixed_images_loaded:true, displayed_values_match_audited_JSON:true, diagnostics_sha256:sha(bytes), gallery});
  }
  check(!exceptions.length, 'Browser JavaScript exception');
  receipt.passed = true;
} catch (error) {
  receipt.error = String(error); throw error;
} finally {
  receipt.browser_exceptions = exceptions;
  await fs.writeFile(path.join(output, 'browser_verification.json'), JSON.stringify(receipt, null, 2) + '\n', {flag: 'wx'});
  ws.close(); await fetch(debug + '/json/close/' + tab.id);
  process.stdout.write(JSON.stringify({passed:receipt.passed,viewports:receipt.viewports,error:receipt.error}) + '\n');
}
