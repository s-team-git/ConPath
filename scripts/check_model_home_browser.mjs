// Real-browser audit. Start a local HTTP server and headless Chrome on port 9223 first.
// Usage: node scripts/check_model_home_browser.mjs [site-url] [output-directory]
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';

const base=process.argv[2] || 'http://127.0.0.1:8766/';
const output=process.argv[3] || 'results/model_home_publication_20260909/home_browser';
await fs.mkdir(output,{recursive:true});
const tab=await fetch('http://127.0.0.1:9223/json/new?about:blank',{method:'PUT'}).then(r=>r.json());
const ws=new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let nextId=0;const pending=new Map();const errors=[];
ws.onmessage=({data})=>{const value=JSON.parse(data);if(value.id){const p=pending.get(value.id);pending.delete(value.id);value.error?p?.reject(value.error):p?.resolve(value.result);}else if(value.method==='Runtime.exceptionThrown'){errors.push(value.params.exceptionDetails);}};
const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
const evaluate=async(expression)=>{const result=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
const delay=(ms)=>new Promise(resolve=>setTimeout(resolve,ms));
const check=(value,message)=>{if(!value)throw new Error(message);};
async function ready(){for(let i=0;i<80;i++){if(await evaluate('document.documentElement?.dataset?.homeReady === "true"'))return;await delay(100);}throw new Error('Interactive page did not initialize');}
async function images(){await evaluate(`(async()=>{await Promise.all([...document.images].filter(i=>i.getAttribute('src') && i.getClientRects().length).map(async i=>{i.loading='eager';try{await i.decode()}catch{}}));})()`);}
async function screenshot(name){await images();const shot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(output,name+'.png'),Buffer.from(shot.data,'base64'));}
async function status(){return await evaluate(`({lang:document.documentElement.lang,width:innerWidth,scrollWidth:document.documentElement.scrollWidth,rows:document.querySelectorAll('[data-home-method]').length,broken:[...document.images].filter(i=>i.getAttribute('src') && i.getClientRects().length && (!i.complete || !i.naturalWidth)).map(i=>i.src),errorBanner:!document.querySelector('#home-error').hidden})`);}
const selectCase = async (index, model, view) => {
  await evaluate(`document.querySelector('[data-case="${index}"]').click();document.querySelector('#home-model').value=${JSON.stringify(model)};document.querySelector('#home-model').dispatchEvent(new Event('change'));document.querySelector('[data-view="${view}"]').click()`);
  await images();
};

const reports=[];
try {
  await command('Page.enable');await command('Runtime.enable');await command('Network.enable');
  await command('Network.setCacheDisabled',{cacheDisabled:true});
  for (const width of [1440,390]) {
    await command('Emulation.setDeviceMetricsOverride',{width,height:width===1440?1100:844,deviceScaleFactor:1,mobile:width<700});
    await command('Page.navigate',{url:base});await ready();await images();
    const data = await evaluate(`JSON.parse(document.querySelector('#home-data').textContent)`);
    const options = await evaluate(`[...document.querySelector('#home-model').options].map(option=>option.value)`);
    const methods = data.result_methods || ['correlated','independent','tiny_deterministic','all_floor'];
    const cases = data.examples;
    check(cases.length > 0, 'No published model cases');
    const expectedModels = [...new Set(cases.flatMap(row=>Object.keys(row.event_probability).filter(model=>row.panels[model+'_sample'] && row.panels[model])))].sort();
    check(JSON.stringify([...options].sort())===JSON.stringify(expectedModels),'Model selector omitted an available model');
    const availableModels = row => options.filter(model => row.panels[model+'_sample'] && row.panels[model] && Number.isFinite(row.event_probability[model]));
    const modelViewStates = cases.reduce((count,row)=>count+availableModels(row).length*2,0);
    const coherent = data.results_cohort === 'coherent_matched_two_seed';
    const hasPilot = coherent || data.results_cohort === 'new_pilot';
    const resultSource = coherent ? 'data/coherent_parent_pilot_zh.json' : hasPilot ? 'data/parent_group_pilot_zh.json' : 'data/current_baseline_k4_analysis.json';
    const scoreText = await fetch(new URL(resultSource, base)).then(response=>{check(response.ok,'Published results unavailable');return response.text();});
    const scores = JSON.parse(scoreText);
    await screenshot(`${width}-home`);
    check(await evaluate(`document.querySelectorAll('#output-gallery figure').length===3 && document.querySelectorAll('[data-case]').length===${cases.length} && JSON.stringify([...document.querySelectorAll('[data-home-method]')].map(row=>row.dataset.homeMethod))===${JSON.stringify(JSON.stringify(methods))}`),'Homepage structure mismatch');
    const resultRows = await evaluate(`[...document.querySelectorAll('[data-home-method]')].map(row=>({method:row.dataset.homeMethod,values:[...row.querySelectorAll('td')].map(cell=>cell.textContent)}))`);
    for (const row of resultRows) {
      const budget = ['correlated','independent'].includes(row.method) ? '32' : '1';
      const score = hasPilot && !coherent ? scores.methods[row.method].budgets[budget] : scores.methods[row.method];
      const samples = hasPilot && !coherent ? (row.method === 'train_radius_prior' ? '—' : budget) : String(score.samples);
      check(JSON.stringify(row.values)===JSON.stringify([samples,score.brier.mean.toFixed(4),(score.risk30.mean*100).toFixed(2)+'%']),'Summary table mixed cohorts or mismatched published metrics: '+row.method);
    }
    if (hasPilot) {
      const baselineText = coherent ? await fetch(new URL('data/parent_group_pilot_zh.json',base)).then(response=>response.text()) : scoreText;
      const verification = await fetch(new URL('data/parent_group_pilot_verification.json',base)).then(response=>response.json());
      check(verification.passed===true && verification.analysis_sha256===createHash('sha256').update(baselineText).digest('hex') && data.pilot_publication_state==='verified','Pilot appeared before verification or result checksum changed');
      const pilotCount = cases.filter(row=>row.cohort==='new_pilot').length;
      check(pilotCount===data.pilot_cases && cases.slice(0,pilotCount).every(row=>row.cohort==='new_pilot') && cases.slice(pilotCount).every(row=>row.cohort!=='new_pilot'),'Pilot cases are not placed before historical cases');
      check(await evaluate(`document.querySelector('#results a[href="pilot.html#scores"]') && document.querySelector('#results').textContent.includes('非最终测试')`),'Pilot scope or report link absent');
    }
    if (coherent) {
      const verification = await fetch(new URL('data/coherent_parent_pilot_verification.json',base)).then(response=>response.json());
      const galleryVerification = await fetch(new URL('data/coherent_pilot_gallery_verification.json',base)).then(response=>response.json());
      const galleryText = await fetch(new URL('data/coherent_pilot_gallery_zh.json',base)).then(response=>response.text());
      const digest = createHash('sha256').update(scoreText).digest('hex');
      check(verification.passed===true && verification.analysis_sha256===digest && data.coherent_publication_state==='verified','Improvement score audit or hash absent');
      check(galleryVerification.passed===true && galleryVerification.analysis_sha256===digest && galleryVerification.gallery_sha256===createHash('sha256').update(galleryText).digest('hex'),'Improvement image audit or hash absent');
      check(JSON.stringify(scores.training_seeds)==='[20260910,20260911]' && JSON.stringify(data.result_seeds)===JSON.stringify(scores.training_seeds) && scores.final_test===false && scores.validation_reused_for_model_development===true,'Improvement seed matching or development scope changed');
      check(methods.length===4 && methods.every(method=>JSON.stringify(scores.methods[method].seeds)===JSON.stringify(scores.training_seeds)),'A result method includes unmatched seeds');
      check(cases.length===40 && data.coherent_cases===10 && cases.slice(0,10).every(row=>row.panels.coherent_categorical_sample && row.seed===20260910 && row.samples===32) && cases.slice(10).every(row=>!row.panels.coherent_categorical_sample),'Improvement duplicated or changed the fixed cases');
      check(await evaluate(`document.querySelector('#home-model').value==='coherent_categorical' && document.querySelector('#results a[href="coherent.html#scores"]') && document.querySelector('#results').textContent.includes('开发复用') && document.querySelector('#results').textContent.includes('区间仍跨零') && document.querySelector('#next').textContent.includes('当前不追加训练')`),'Improvement defaults, report link, or Chinese interpretation missing');
      check(await evaluate(`(()=>{const d=JSON.parse(document.querySelector('#home-data').textContent);return [...document.querySelectorAll('[data-case] img')].every((image,i)=>image.getAttribute('src')===(d.examples[i].panels.coherent_categorical_sample||d.examples[i].panels.correlated_sample));})()`),'Thumbnails do not show the available model version');
    }
    check(await evaluate(`!document.querySelector('#training-ablations') && !document.querySelector('#flatlands-external-progress')`),'Research detail leaked into homepage');
    await evaluate(`document.querySelector('#effects').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-effects`);
    for (const [index,row] of cases.entries()) for (const model of availableModels(row)) for (const view of ['sample','probability']) {
      await selectCase(index,model,view);
      const predictionKey = model+(view==='sample'?'_sample':'');
      const failure = (row.event_probability[model]>=.5)!==Boolean(row.target);
      check(await evaluate(`(()=>{const d=JSON.parse(document.querySelector('#home-data').textContent).examples[${index}];return document.querySelector('#image-prediction').getAttribute('src')===d.panels[${JSON.stringify(predictionKey)}] && document.querySelector('#image-input').getAttribute('src')===d.panels.observed && document.querySelector('#image-reference').getAttribute('src')===d.panels.reference && document.querySelector('#case-probability').textContent===(d.event_probability[${JSON.stringify(model)}]*100).toFixed(1)+'%';})()`),'Rendered model or probability mismatches frozen source');
      check(await evaluate(`document.querySelector('#case-reading').classList.contains('failure')===${failure} && document.querySelector('#case-explanation').textContent.includes(${JSON.stringify(failure?'预测失败':'预测正确')})`),'Chinese outcome explanation disagrees with reference target');
      if (row.panels.coherent_categorical) {
        const label = {coherent_categorical:'ConPath改进版',correlated:'原ConPath',independent:'独立单元对照'}[model];
        check(await evaluate(`document.querySelector('[data-panel="prediction"] figcaption span').textContent.includes(${JSON.stringify(label)}) && document.querySelector('#case-source').textContent.includes('相同种子20260910') && document.querySelector('#case-source').textContent.includes('开发复用') && document.querySelector('#case-source').textContent.includes('第1次采样') && document.querySelector('#case-source').textContent.includes('32次采样')`),'Model version, seed, sample identity, or development reuse caption absent');
      }
      check((await status()).broken.length===0,'Broken model map');
    }
    const failureCase = cases.flatMap((row,index)=>availableModels(row).map(model=>({row,index,model}))).find(({row,model})=>(row.event_probability[model]>=.5)!==Boolean(row.target));
    check(failureCase,'Published cases contain no auditable failure');
    await selectCase(failureCase.index,failureCase.model,'sample');
    check(await evaluate(`document.querySelector('#case-reading').classList.contains('failure') && document.querySelector('#case-explanation').textContent.includes('失败')`),'Failure explanation absent');
    if(width<700){
      await evaluate(`document.querySelector('[data-jump="1"]').click()`);await delay(650);
      check(await evaluate(`(()=>{const gallery=document.querySelector('#output-gallery');const expected=gallery.children[1].offsetLeft-gallery.children[0].offsetLeft;return Math.abs(gallery.scrollLeft-expected)<5 && document.querySelector('[data-jump="1"]').getAttribute('aria-pressed')==='true';})()`),'Mobile prediction navigation failed');
    }
    await evaluate(`document.querySelector('#output-gallery').scrollIntoView({behavior:'instant',block:'center'})`);await screenshot(`${width}-failure`);
    await evaluate(`document.querySelector('#image-prediction').closest('a').click()`);await images();
    check(await evaluate(`document.querySelector('#model-dialog').open && document.querySelector('#model-dialog-image').src===document.querySelector('#image-prediction').src`),'Model image zoom mismatch');
    await screenshot(`${width}-zoom`);
    await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    check(await evaluate(`!document.querySelector('#model-dialog').open`),'Escape close failed');
    await evaluate(`document.querySelector('#results').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-results`);
    await evaluate(`document.querySelector('#next').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-next`);
    const sources = [...new Set(cases.map(row=>row.source))].sort();
    check(await evaluate(`JSON.stringify([...document.querySelector('#home-source').options].map(option=>option.value).filter(value=>value!=='all').sort())===${JSON.stringify(JSON.stringify(sources))}`),'Source choices do not match published data');
    for (const source of sources) {
      const ids = cases.map((row,index)=>({row,index})).filter(({row})=>row.source===source).map(({index})=>index);
      await evaluate(`document.querySelector('#home-source').value=${JSON.stringify(source)};document.querySelector('#home-source').dispatchEvent(new Event('change'))`);
      check(await evaluate(`document.querySelector('#case-counter').textContent===${JSON.stringify('1 / '+ids.length)} && JSON.stringify([...document.querySelectorAll('[data-case]')].filter(button=>!button.hidden).map(button=>Number(button.dataset.case)))===${JSON.stringify(JSON.stringify(ids))}`),'Source filter or count mismatch: '+source);
      for (const [button,expectedIndex] of [['case-prev',ids.at(-1)],['case-next',ids[0]],['case-next',ids[1%ids.length]]]) {
        await evaluate(`document.querySelector('#${button}').click()`);await images();
        check(await evaluate(`document.querySelector('[data-case="${expectedIndex}"]').getAttribute('aria-pressed')==='true'`),'Filtered paging or wraparound failed: '+source);
      }
    }
    await evaluate(`document.querySelector('#home-source').value='all';document.querySelector('#home-source').dispatchEvent(new Event('change'))`);
    for (const [index,row] of cases.entries()) {
      const available = availableModels(row);
      for (const model of options.filter(value=>!available.includes(value))) {
        await selectCase(index,model,'sample');
        check(await evaluate(`document.querySelector('#home-model').selectedOptions[0].disabled===false && document.querySelector('#home-model option[value="${model}"]').disabled`),'Unavailable model did not fall back to an available model');
        if (model==='coherent_categorical') check(await evaluate(`document.querySelector('#home-model').value==='correlated' && document.querySelector('#case-source').textContent.includes('已显示原ConPath')`),'Historical case lacks explicit Chinese model fallback');
      }
    }
    if (coherent) {
      await selectCase(10,'coherent_categorical','sample');
      await evaluate(`document.querySelector('[data-case="0"]').click()`);
      check(await evaluate(`document.querySelector('#home-model').value==='coherent_categorical'`),'Preferred improvement model was lost after visiting an older case');
    }
    await images();const report=await status();check(report.scrollWidth<=width+1 && report.rows===methods.length && !report.errorBanner && !report.broken.length,'Homepage viewport or runtime failure: '+JSON.stringify(report));
    reports.push({...report,cases:cases.length,modelViewStates,panels:3,sources:sources.length,resultsCohort:data.results_cohort||'historical',firstActualWorldAlwaysShown:true});
    if (hasPilot) {
      await command('Page.navigate',{url:new URL('pilot.html',base).href});
      for(let i=0;i<80;i++){if(await evaluate(`document.querySelectorAll('#scores tbody tr').length===7`))break;await delay(100);}
      check(await evaluate(`document.querySelectorAll('#scores tbody tr').length===7 && document.documentElement.lang==='zh-CN'`),'Detailed pilot report is missing');
      await images();
      await evaluate(`document.querySelector('#scores').scrollIntoView({behavior:'instant',block:'start'})`);
      await screenshot(`${width}-pilot-report`);
      await evaluate(`document.querySelector('details').open=true`);await images();
      const pilot = await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,broken:[...document.images].filter(i=>!i.complete || !i.naturalWidth).map(i=>i.src),pdfLinks:document.querySelectorAll('a[href$=".pdf"]').length})`);
      check(pilot.scrollWidth<=width+1 && !pilot.broken.length && pilot.pdfLinks===2,'Detailed pilot report layout or images failed: '+JSON.stringify(pilot));
      reports.at(-1).detailedPilotReport = pilot;
    }
    if (coherent) {
      await command('Page.navigate',{url:new URL('coherent.html',base).href});
      for(let i=0;i<80;i++){if(await evaluate(`document.querySelectorAll('#scores tbody tr').length===4`))break;await delay(100);}
      check(await evaluate(`document.querySelectorAll('#scores tbody tr').length===4 && document.documentElement.lang==='zh-CN' && document.body.textContent.includes('20260910') && document.body.textContent.includes('20260911') && document.body.textContent.includes('非最终测试')`),'Detailed improvement report or paired-seed scope missing');
      await images();
      await evaluate(`document.querySelector('#scores').scrollIntoView({behavior:'instant',block:'start'})`);
      await screenshot(`${width}-coherent-report`);
      await evaluate(`document.querySelectorAll('details').forEach(detail=>detail.open=true)`);await images();
      const detail = await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,broken:[...document.images].filter(i=>!i.complete || !i.naturalWidth).map(i=>i.src),pdfLinks:document.querySelectorAll('a[href$=".pdf"]').length})`);
      check(detail.scrollWidth<=width+1 && !detail.broken.length && detail.pdfLinks===2,'Detailed improvement report layout or charts failed: '+JSON.stringify(detail));
      reports.at(-1).detailedImprovementReport = detail;
    }
  }
  await command('Page.navigate',{url:base+'#baseline-review'});
  for(let i=0;i<80;i++){if(await evaluate(`location.pathname.endsWith('/research.html') && location.hash==='#baseline-review'`))break;await delay(100);}
  check(await evaluate(`location.pathname.endsWith('/research.html') && location.hash==='#baseline-review'`),'Historical bookmark was not redirected');
  check(errors.length===0,'JavaScript exceptions');
  const report={passed:true,url:base,viewports:reports,legacyBookmarkRedirectPassed:true,errors};
  await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}finally{ws.close();await fetch(`http://127.0.0.1:9223/json/close/${tab.id}`);}
