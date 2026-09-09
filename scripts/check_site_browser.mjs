// Real-browser audit. Start a local HTTP server and headless Chrome on port 9223 first.
// Usage: node scripts/check_site_browser.mjs [site-url] [output-directory]
import fs from 'node:fs/promises';
import path from 'node:path';

const base=process.argv[2] || 'http://127.0.0.1:8765/';
const output=process.argv[3] || 'results/site_redesign_20260907/browser';
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
async function ready(){for(let i=0;i<80;i++){if(await evaluate('document.documentElement?.dataset?.interactiveReady === "true"'))return;await delay(100);}throw new Error('Interactive page did not initialize');}
async function images(){await evaluate(`(async()=>{await Promise.all([...document.images].filter(i=>i.getAttribute('src') && i.getClientRects().length).map(async i=>{i.loading='eager';try{await i.decode()}catch{}}));})()`);}
async function screenshot(name){await images();const shot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(output,name+'.png'),Buffer.from(shot.data,'base64'));}
async function status(){return await evaluate(`({lang:document.documentElement.lang,width:innerWidth,scrollWidth:document.documentElement.scrollWidth,rows:document.querySelectorAll('.metrics-table tbody tr').length,broken:[...document.images].filter(i=>i.getAttribute('src') && i.getClientRects().length && (!i.complete || !i.naturalWidth)).map(i=>i.src),errorBanner:!document.querySelector('#interaction-error').hidden})`);}

const reports=[];
try{
  await command('Page.enable');await command('Runtime.enable');
  await command('Network.enable');await command('Network.setCacheDisabled',{cacheDisabled:true});
  for(const width of [1440,390]){
    await command('Emulation.setDeviceMetricsOverride',{width,height:width===1440?1100:844,deviceScaleFactor:1,mobile:width<600});
    await command('Page.navigate',{url:base});await ready();await images();
    check(await evaluate(`document.querySelector('#example-probability').textContent.includes('78.9%')`),'Initial example probability mismatch');
    await screenshot(`${width}-home`);
    await evaluate(`document.querySelector('#method').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-method`);
    if(width<600){await evaluate(`document.querySelector('[data-step="3"]').click()`);await delay(650);check(await evaluate(`document.querySelector('#example-panels').scrollLeft>500`),'Mobile step navigation failed');}
    await evaluate(`document.querySelector('[data-example="1"]').click();document.querySelector('#model-select').value='independent';document.querySelector('#model-select').dispatchEvent(new Event('change'))`);
    check(await evaluate(`document.querySelector('#example-probability').textContent.includes('80.5%') && document.querySelector('#example-explanation').textContent.includes('失败')`),'Failure example or model switch mismatch');
    await evaluate(`document.querySelector('#example-panels [data-zoom]').click()`);
    check(await evaluate(`document.querySelector('#image-dialog').open`),'Image dialog did not open');
    await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    check(await evaluate(`!document.querySelector('#image-dialog').open`),'Escape did not close dialog');
    for(const dataset of ['flatlands','unscenes3d']){
      await evaluate(`document.querySelector('[data-dataset="${dataset}"]').click();document.querySelector('#datasets').scrollIntoView({behavior:'instant',block:'start'})`);
      for(let index=0;index<6;index++){
        await evaluate(`document.querySelector('[data-gallery-index="${index}"]').click()`);await images();
        check((await status()).broken.length===0,`${dataset} scene ${index} has broken images`);
      }
    }
    await screenshot(`${width}-gallery`);
    await evaluate(`document.querySelector('#sequence-stage').scrollIntoView({behavior:'instant',block:'center'});document.querySelector('#sequence-frame').value='17';document.querySelector('#sequence-frame').dispatchEvent(new Event('input'))`);
    check(await evaluate(`document.querySelector('#sequence-counter').textContent==='18 / 18'`),'Frame slider mismatch');
    await screenshot(`${width}-sequence`);
    await evaluate(`document.querySelector('#sequence-play').click()`);await delay(1250);
    check(await evaluate(`Number(document.querySelector('#sequence-frame').value)>0`),'Sequence playback did not advance');
    await evaluate(`document.querySelector('#sequence-play').click()`);
    check(await evaluate(`document.querySelector('#sequence-play').getAttribute('aria-pressed')==='false'`),'Sequence pause failed');
    await evaluate(`document.querySelector('#results').scrollIntoView({behavior:'instant',block:'start'})`);
    for(const chart of ['brier','risk','reliability','sampling','dependence']){
      await evaluate(`document.querySelector('button[data-chart="${chart}"]').click()`);await images();
      check(await evaluate(`document.querySelector('#chart-image').getAttribute('src').endsWith('/${chart}.svg') && document.querySelector('#chart-pdf').getAttribute('href').endsWith('/${chart}.pdf')`),'Chart/PDF switch mismatch');
      check((await status()).broken.length===0,'Broken chart');
    }
    await evaluate(`document.querySelector('button[data-chart="brier"]').click();document.querySelector('.chart-toolbar').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-chart`);
    check(await evaluate(`document.querySelectorAll('[data-ablation]').length===3`),'Missing completed ablation rows');
    await evaluate(`document.querySelector('#training-ablations').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-ablations`);
    await evaluate(`document.querySelector('#ablation-details').open=true`);await images();
    check(await evaluate(`[...document.querySelectorAll('.ablation-chart img')].every(i=>i.currentSrc.endsWith('${width<600?'-mobile':''}.svg'))`),'Ablation responsive chart source mismatch');
    await evaluate(`document.querySelector('#ablation-details .ablation-chart').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-ablation-intervals`);
    await evaluate(`document.querySelector('#ablation-details .ablation-chart [data-zoom]').click()`);await images();
    check(await evaluate(`document.querySelector('#image-dialog').open && document.querySelector('#dialog-image').src.endsWith('training-ablation-paired${width<600?'-mobile':''}.svg')`),'Ablation zoom source mismatch');
    await evaluate(`document.querySelector('#dialog-close').click()`);
    check((await status()).broken.length===0,'Broken ablation assets');
    await evaluate(`document.querySelector('#ablation-details').open=false`);
    await evaluate(`document.querySelector('#ablation-examples').open=true;document.querySelector('#ablation-examples').scrollIntoView({behavior:'instant',block:'start'})`);await images();
    check(await evaluate(`document.querySelectorAll('.ablation-map-panels img').length===6`),'Missing actual ablation case images');
    check((await status()).broken.length===0,'Broken ablation case image');await screenshot(`${width}-ablation-cases`);
    await evaluate(`document.querySelector('[data-ablation-view="footprint"]').click()`);await images();
    check(await evaluate(`[...document.querySelectorAll('.ablation-map-panels img')].every(i=>i.src.endsWith('-footprint.svg') && i.alt.includes('机器人'))`),'Footprint sample view mismatch');
    check((await status()).broken.length===0,'Broken footprint sample image');await screenshot(`${width}-ablation-footprints`);
    await evaluate(`document.querySelector('[data-ablation-view="probability"]').click()`);await images();
    check(await evaluate(`[...document.querySelectorAll('.ablation-map-panels img')].every(i=>!i.src.endsWith('-footprint.svg'))`),'Probability map switch failed');
    await evaluate(`document.querySelector('#ablation-examples').open=false`);
    if(await evaluate(`Boolean(document.querySelector('#external-progress'))`)){
      check(await evaluate(`document.querySelectorAll('.native-case').length===3 && document.querySelectorAll('.native-map-panels img').length===21`),'Missing native sanity cases');
      for(let index=0;index<3;index++){
        await evaluate(`{const c=document.querySelectorAll('.native-case')[${index}];c.open=true;c.querySelector('details').open=true;c.scrollIntoView({behavior:'instant',block:'start'});}`);await images();
        check(await evaluate(`getComputedStyle(document.querySelectorAll('.native-case')[${index}].querySelector('.native-map-panels')).gridTemplateColumns.split(' ').length===${width<600?1:3}`),'Native responsive column count mismatch');
        check((await status()).broken.length===0,'Broken native output image');
        if(index===0){
          await screenshot(`${width}-native-case`);
          await evaluate(`document.querySelector('.native-case .native-map-panels a:nth-child(2)').click()`);await images();
          check(await evaluate(`document.querySelector('#image-dialog').open && document.querySelector('#dialog-image').src.endsWith('room-vote.png')`),'Native vote zoom failed');
          await screenshot(`${width}-native-vote`);
          await evaluate(`document.querySelector('#dialog-close').click()`);
        }
        check((await status()).scrollWidth<=width+1,'Native gallery overflows viewport');
        await evaluate(`document.querySelectorAll('.native-case')[${index}].open=false`);
      }
    }
    if(await evaluate(`Boolean(document.querySelector('#flatlands-external-progress'))`)){
      check(await evaluate(`document.querySelectorAll('.external-case').length===2 && document.querySelectorAll('.external-map-panels img').length===16`),'Missing FlatLands external diagnostic images');
      await evaluate(`document.querySelector('#flatlands-external-progress').scrollIntoView({behavior:'instant',block:'start'})`);await screenshot(`${width}-external-cost`);
      for(let index=0;index<2;index++){
        await evaluate(`{const c=document.querySelectorAll('.external-case')[${index}];c.open=true;c.querySelector('details').open=true;c.scrollIntoView({behavior:'instant',block:'start'});}`);await images();
        check(await evaluate(`getComputedStyle(document.querySelectorAll('.external-case')[${index}].querySelector('.external-map-panels')).gridTemplateColumns.split(' ').length===${width<600?1:4}`),'External responsive column count mismatch');
        check((await status()).broken.length===0,'Broken external diagnostic image');
        if(index===0){
          await screenshot(`${width}-external-case`);
          await evaluate(`document.querySelector('.external-case .external-map-panels a:nth-child(3)').click()`);await images();
          check(await evaluate(`document.querySelector('#image-dialog').open && document.querySelector('#dialog-image').src.endsWith('case0-flow.png')`),'External vote zoom failed');
          await screenshot(`${width}-external-vote`);await evaluate(`document.querySelector('#dialog-close').click()`);
          await evaluate(`document.querySelector('.external-case details .external-map-panels a').click()`);await images();
          check(await evaluate(`document.querySelector('#image-dialog').open && document.querySelector('#dialog-image').src.endsWith('case0-flow_world0.png')`),'External full-map sample zoom failed');
          await evaluate(`document.querySelector('#dialog-close').click()`);
        }
        check((await status()).scrollWidth<=width+1,'External gallery overflows viewport');
        await evaluate(`document.querySelectorAll('.external-case')[${index}].open=false`);
      }
      await evaluate(`{const d=document.querySelector('#flatlands-external-progress > details:last-of-type');d.open=true;d.scrollIntoView({behavior:'instant',block:'start'});}`);await images();
      check((await status()).broken.length===0,'Broken external training curves');await screenshot(`${width}-external-curves`);
      await evaluate(`document.querySelector('#flatlands-external-progress > details:last-of-type').open=false`);
    }
    const report=await status();check(report.lang==='zh-CN' && report.rows===9,'Language or result row count mismatch');check(report.scrollWidth<=report.width+1,'Page overflows viewport');check(!report.errorBanner && report.broken.length===0,'Runtime/image error');
    reports.push({...report,ablationRows:3,ablationCaseImages:6,interactions:['both examples','both models','dialog open/Escape','12 gallery scenes','frame seek/play/pause','five charts with PDF links','ablation table','two responsive ablation charts','ablation zoom','two fixed cases with three actual model maps each','three native CogniPlan cases and zoom','two external cases, 16 labelled images, vote/sample zoom','external cost table and two training curves']});
  }
  check(errors.length===0,'JavaScript exceptions');
  await fs.writeFile(path.join(output,'report.json'),JSON.stringify({passed:true,url:base,viewports:reports,errors},null,2)+'\n');
  console.log(JSON.stringify({passed:true,viewports:reports,errors},null,2));
}finally{ws.close();await fetch(`http://127.0.0.1:9223/json/close/${tab.id}`);}
