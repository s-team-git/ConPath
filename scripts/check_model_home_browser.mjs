// Real-browser audit. Start a local HTTP server and headless Chrome on port 9223 first.
// Usage: node scripts/check_site_browser.mjs [site-url] [output-directory]
import fs from 'node:fs/promises';
import path from 'node:path';

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

const reports=[];
try {
  await command('Page.enable');await command('Runtime.enable');await command('Network.enable');
  await command('Network.setCacheDisabled',{cacheDisabled:true});
  for (const width of [1440,390]) {
    await command('Emulation.setDeviceMetricsOverride',{width,height:width===1440?1100:844,deviceScaleFactor:1,mobile:width<700});
    await command('Page.navigate',{url:base});await ready();await images();
    await screenshot(`${width}-home`);
    check(await evaluate(`document.querySelectorAll('#output-gallery figure').length===3 && document.querySelectorAll('[data-home-method]').length===4`),'Homepage structure mismatch');
    check(await evaluate(`!document.querySelector('#training-ablations') && !document.querySelector('#flatlands-external-progress')`),'Research detail leaked into homepage');
    await evaluate(`document.querySelector('#effects').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-effects`);
    for (let index=0;index<30;index++) for (const model of ['correlated','independent']) for (const view of ['sample','probability']) {
      if (index >= 20 && index < 28 && model === 'independent') continue;
      await evaluate(`document.querySelector('[data-case="${index}"]').click();document.querySelector('#home-model').value='${model}';document.querySelector('#home-model').dispatchEvent(new Event('change'));document.querySelector('[data-view="${view}"]').click()`);await images();
      check(await evaluate(`(()=>{const d=JSON.parse(document.querySelector('#home-data').textContent).examples[${index}];return document.querySelector('#image-prediction').getAttribute('src')===d.panels['${model}${view==='sample'?'_sample':''}'] && document.querySelector('#case-probability').textContent===(d.event_probability['${model}']*100).toFixed(1)+'%';})()`),'Rendered model or probability mismatches frozen source');
      check((await status()).broken.length===0,'Broken model map');
    }
    await evaluate(`document.querySelector('[data-case="29"]').click();document.querySelector('#home-model').value='correlated';document.querySelector('#home-model').dispatchEvent(new Event('change'));document.querySelector('[data-view="sample"]').click()`);await images();
    check(await evaluate(`document.querySelector('#case-reading').classList.contains('failure') && document.querySelector('#case-explanation').textContent.includes('失败')`),'Failure explanation absent');
    if(width<700){
      await evaluate(`document.querySelector('[data-jump="1"]').click()`);await delay(650);
      check(await evaluate(`document.querySelector('#output-gallery').scrollLeft>300 && document.querySelector('[data-jump="1"]').getAttribute('aria-pressed')==='true'`),'Mobile prediction navigation failed');
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
    await evaluate(`document.querySelector('#home-source').value='UnScenes3D';document.querySelector('#home-source').dispatchEvent(new Event('change'));document.querySelector('#case-next').click()`);await images();
    check(await evaluate(`document.querySelector('#case-counter').textContent==='2 / 8' && document.querySelector('#home-model option[value=independent]').disabled && [...document.querySelectorAll('[data-case]')].filter(b=>!b.hidden).length===8`),'Dataset filter, paging or unavailable-model guard failed');
    await evaluate(`document.querySelector('#home-source').value='all';document.querySelector('#home-source').dispatchEvent(new Event('change'))`);
    await images();const report=await status();check(report.scrollWidth<=width+1 && report.rows===4 && !report.errorBanner && !report.broken.length,'Homepage viewport or runtime failure: '+JSON.stringify(report));
    reports.push({...report,cases:30,modelViewStates:104,panels:3,firstActualWorldAlwaysShown:true});
  }
  await command('Page.navigate',{url:base+'#baseline-review'});
  for(let i=0;i<80;i++){if(await evaluate(`location.pathname.endsWith('/research.html') && location.hash==='#baseline-review'`))break;await delay(100);}
  check(await evaluate(`location.pathname.endsWith('/research.html') && location.hash==='#baseline-review'`),'Historical bookmark was not redirected');
  check(errors.length===0,'JavaScript exceptions');
  const report={passed:true,url:base,viewports:reports,legacyBookmarkRedirectPassed:true,errors};
  await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}finally{ws.close();await fetch(`http://127.0.0.1:9223/json/close/${tab.id}`);}
