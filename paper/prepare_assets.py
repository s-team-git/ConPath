#!/usr/bin/env python3
"""Render manuscript tables from existing summaries only; never load a model/dataset."""
from pathlib import Path
import csv
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def esc(s):
    s = s.strip().replace('**', '')
    m = {'&':r'\&','%':r'\%','_':r'\_','#':r'\#',
         '±':r'$\pm$','↓':r'$\downarrow$','↑':r'$\uparrow$',
         '—':'---','−':'-'}
    return ''.join(m.get(c,c) for c in s)

def render_table(header,rows):
    return ('% Generated from immutable reported values; no model inference.\n'
            r'\resizebox{\linewidth}{!}{\begin{tabular}{@{}l'+
            'r'*(len(header)-1)+r'@{}}\toprule'+'\n'+
            ' & '.join(map(esc,header))+r'\\\midrule'+'\n'+
            '\n'.join(' & '.join(map(esc,row))+r'\\' for row in rows)+
            '\n'+r'\bottomrule\end{tabular}}'+'\n')

def main():
    td=HERE/'tables';td.mkdir(exist_ok=True)
    text=(ROOT/'PAPER_TABLES.md').read_text()
    records=[]
    for key in ['T1','T1b','T1c','T1d','T1e','T2','T3','T4']:
        section=re.search(r'^## '+key+r'\. (.*?)(?=^## |\Z)',text,re.M|re.S)
        if not section: raise ValueError(key)
        raw=re.search(r'^\|.*(?:\n\|.*)+',section.group(),re.M).group()
        lines=[x.strip().strip('|').split('|') for x in raw.splitlines()]
        header=[x.strip() for x in lines[0]]
        rows=[[x.strip() for x in row] for row in lines[2:]]
        if key=='T2':
            header[2]='Seed count'
            for row in rows: row[2]='1' if row[0]=='Direct-query' else '3'
        (td/(key+'.tex')).write_text(render_table(header,rows))
        records.append({'id':key,'source':'PAPER_TABLES.md','source_sha256':sha(ROOT/'PAPER_TABLES.md'),
                        'section':section.group().splitlines()[0],'rows':rows,'header':header,
                        'generated_path':str((td/(key+'.tex')).relative_to(ROOT))})
    source=ROOT/'results/paper_submission_v1/analysis/primary_three_seed_strata.csv'
    if source.exists():
        rows=[]
        names={'correlated':'ConPath','independent':'Independent','deterministic':'Deterministic'}
        for row in csv.DictReader(source.open()):
            if row['stratum_dimension']!='radius': continue
            rows.append([names[row['method']],row['K'],row['stratum_value'],
                         f"{float(row['brier_mean']):.5f} ± {float(row['brier_sample_sd']):.5f}",
                         f"{float(row['ece_mean']):.5f}",
                         f"{float(row['false_safe_at_0_8_mean']):.5f}" if row['false_safe_at_0_8_mean'] else 'Undefined',
                         f"{float(row['coverage_at_0_8_mean']):.5f}"])
        header=['Method','K','Radius (cells)','Event Brier ↓','ECE ↓','False-safe @0.8','Coverage @0.8']
        (td/'radius.tex').write_text(render_table(header,rows))
        records.append({'id':'radius','source':str(source.relative_to(ROOT)), 'source_sha256':sha(source),'header':header,'rows':rows})
    sources=['PAPER_DRAFT.md','PAPER_TABLES.md','PAPER_FIGURES.md','PAPER_EVIDENCE.md',
             'results/paper_validation_snapshot.json','results/FINAL_MODEL_SELECTION.md']
    out={'source_documents':[{'path':x,'sha256':sha(ROOT/x)} for x in sources],
         'tables':records,'new_training':False,'new_inference':False,'raw_data_or_test_reads':False,
         'rule':'Display-only conversion of existing saved summaries; no result selection.'}
    (HERE/'table_provenance.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print('Rendered',len(records),'tables from saved summaries.')

if __name__=='__main__': main()
