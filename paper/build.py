#!/usr/bin/env python3
"""Compile the working manuscript and supplement; no ML/data-loading dependencies."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    local=HERE/'tools/tectonic'
    engine=str(local) if local.exists() else shutil.which('tectonic')
    if not engine:
        raise SystemExit('Run python paper/bootstrap.py for the pinned local compiler, or install Tectonic yourself.')
    env=os.environ.copy();env['XDG_CACHE_HOME']=str(HERE/'.cache')
    out=HERE/'build';out.mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(HERE/'prepare_assets.py')],check=True,cwd=HERE)
    reports=[]
    for name in ['main','supplement']:
        command=[engine,'--untrusted','--keep-logs','--keep-intermediates','--outdir','build',name+'.tex']
        result=subprocess.run(command,cwd=HERE,env=env,capture_output=True,text=True)
        (out/(name+'_compiler.txt')).write_text(result.stdout+result.stderr)
        if result.returncode:
            print(result.stdout+result.stderr,file=sys.stderr)
            raise SystemExit(result.returncode)
        log=(out/(name+'.log')).read_text(errors='replace')
        info=subprocess.run(['pdfinfo',str(out/(name+'.pdf'))],capture_output=True,text=True,check=True).stdout
        overfull=re.findall(r'Overfull \\[hv]box[^\n]*',log)
        undefined=re.findall(r'[^\n]*(?:(?:Citation|Reference).*undefined|undefined references|multiply defined)[^\n]*',log,re.I)
        fonts=re.findall(r'[^\n]*(?:Font shape.*undefined|Missing character)[^\n]*',log)
        report={'name':name,'pdf':str((out/(name+'.pdf')).relative_to(HERE)),
                'pdf_sha256':sha(out/(name+'.pdf')),'pages':int(re.search(r'Pages:\s+(\d+)',info).group(1)),
                'overfull_boxes':overfull,'undefined_references_or_labels':undefined,
                'missing_font_or_character_warnings':fonts,
                'underfull_box_count':len(re.findall(r'Underfull \\[hv]box',log)),
                'command':command,'log_sha256':sha(out/(name+'.log'))}
        reports.append(report)
        print(name,report['pages'],'pages;',len(overfull),'overfull boxes;',len(undefined),'undefined references')
    inputs=[]
    for pattern in ['*.tex','*.bib','*.py','tables/*.tex','figures/*.pdf','figures/provenance.json','vendor/*','table_provenance.json']:
        for path in sorted(HERE.glob(pattern)):
            if path.is_file(): inputs.append({'path':str(path.relative_to(HERE)),'sha256':sha(path)})
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'engine':subprocess.check_output([engine,'--version'],text=True).strip(),
            'engine_sha256':sha(Path(engine)),'documents':reports,'inputs':inputs,
            'scientific_status':'validation-only working draft; no final-test or submission-readiness claim',
            'new_training':False,'new_inference':False,'test_data_opened':False,
            'template':'IEEEtran 1.8b conference style; edition-specific page rules not checked',
            'figure_language':'Original Chinese labels retained with English captions; data PDFs byte-identical to existing assets.'}
    (out/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(r['overfull_boxes'] or r['undefined_references_or_labels'] or r['missing_font_or_character_warnings'] for r in reports):
        raise SystemExit('Inspect unresolved layout or reference warnings in build_report.json.')

if __name__=='__main__': main()
