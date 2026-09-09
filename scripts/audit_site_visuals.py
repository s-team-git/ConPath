#!/usr/bin/env python3
"""Independently check exported media provenance and the Chinese result table."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


class Table(HTMLParser):
    def __init__(self,attribute='data-method'):super().__init__();self.attribute=attribute;self.method=None;self.rows={};self.in_cell=False;self.cells=[];self.text=''
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='tr' and self.attribute in attrs:self.method=attrs[self.attribute];self.cells=[]
        if self.method and tag in ('td','th'):self.in_cell=True;self.text=''
    def handle_data(self,data):
        if self.in_cell:self.text+=data
    def handle_endtag(self,tag):
        if self.in_cell and tag in ('td','th'):self.cells.append(self.text);self.in_cell=False
        if self.method and tag=='tr':self.rows[self.method]=self.cells;self.method=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/site_visuals_current_audit.json')
    args=parser.parse_args()
    data=read(ROOT/'site/data/site_visuals_zh.json')
    assert data['language']=='zh-CN' and data['test_evaluated'] is False and data['training_started'] is False
    for name,digest in data['sources'].items():assert sha(ROOT/name)==digest,name
    for name,entry in data['assets'].items():
        path=ROOT/'site'/name
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],name
        if path.suffix=='.svg':
            tree=ET.parse(path)
            if path.name.startswith(('flatlands_','unscenes_','example_')):
                assert len(tree.findall('.//{http://www.w3.org/2000/svg}line'))==0,name
                assert tree.find('{http://www.w3.org/2000/svg}title') is not None
                assert '可通行' in path.read_text() and '范围外' in path.read_text()
    manifest=read(ROOT/'results/unscenes3d_contract_manifest_ground_valid/manifest.json')
    allowed={r['timestamp']:r for r in manifest['records']['train']}
    gallery=data['gallery'];photos=set()
    assert len(gallery['flatlands'])==6 and len(gallery['unscenes3d'])==6 and len(gallery['sequence'])==18
    assert len({r['scene'] for r in gallery['unscenes3d']})==6
    assert len({r['location'] for r in gallery['unscenes3d']})==3
    for row in gallery['unscenes3d']+gallery['sequence']:
        assert row['split']=='train' and row['timestamp'] in allowed and row['location']!='location_6'
        assert allowed[row['timestamp']]['scene_id']==row['scene']
        for name,digest in row['raw_sources'].items():assert sha(ROOT/name)==digest,name
        camera=next(name for name in row['raw_sources'] if name.endswith('.jpg'))
        assert sha(ROOT/'site'/row['camera'])==sha(ROOT/camera)
        photos.add(row['camera'])
    assert len({r['scene'] for r in gallery['sequence']})==1
    stamps=[float(r['timestamp']) for r in gallery['sequence']]
    assert stamps==sorted(stamps) and len(set(stamps))==18
    source=read(ROOT/'site/data/flatlands_k128_clean_candidate.json')
    assert len(data['examples'])==2
    for row,original in zip(data['examples'],source['cases']):
        assert row['global_id']==original['global_id'] and row['candidate_index']==original['candidate_index'] and row['target']==original['target']
        assert row['split']=='validation' and row['visual_sampling_seed']==original['visual_sampling_seed']
        for variant,checkpoint in row['checkpoints'].items():
            assert sha(ROOT/checkpoint['path'])==checkpoint['sha256']
            assert row['event_probability'][variant]==original['event_probability'][variant]['20260831']
    evidence_page = ROOT/'site/research.html'
    table=Table();table.feed(evidence_page.read_text())
    methods=read(ROOT/'site/data/flatlands_clean_paper_analysis.json')['methods']
    assert set(table.rows)==set(methods)
    for key,method in methods.items():
        row=table.rows[key];assert int(row[1])==method['seed_count']
        for index,metric in [(2,'brier'),(3,'nll'),(4,'ece')]:
            values=row[index].split(' ± ');expected=method['aggregate'][metric]
            assert values[0]==f"{expected['mean']:.5f}"
            if expected['sample_sd'] is not None:assert values[1]==f"{expected['sample_sd']:.5f}"
        assert row[5]==f"{method['equal_coverage']['0.3']['mean']*100:.2f}%"
    # Media provenance is independent of a concurrent, explicitly authorized trainer.
    # The manifest's training_started flag describes rendering, not live GPU jobs.
    report={'passed':True,'test_evaluated':False,'training_started_by_audit':False,'assets_hashed':len(data['assets']),'unchanged_camera_images':len(photos),'gallery_scenes':12,'training_sequence_frames':18,'fixed_validation_examples':2,'audited_method_rows':len(table.rows),'source_hashes':len(data['sources'])}
    ablation_path=ROOT/'site/data/flatlands_clean_training_ablations.json'
    if ablation_path.is_file():
        ablations=read(ablation_path)
        visuals=read(ROOT/'site/data/training_ablation_visuals_zh.json')
        assert sha(ablation_path)==visuals['report_sha256']
        assert sha(ablation_path)==sha(ROOT/'results/paper_clean_ablation_matrix_v1/analysis/report.json')
        assert sha(ROOT/'scripts/build_ablation_summary_zh.py')==visuals['builder_sha256']
        assert ablations['test_evaluated'] is False and len(ablations['audits'])==6
        assert all(audit['passed'] for audit in ablations['audits'])
        assert len(visuals['assets'])==6
        for asset in visuals['assets']:
            path=ROOT/'site'/asset['path']
            assert sha(path)==asset['sha256'],asset['path']
            if path.suffix=='.svg':
                ET.parse(path)
                text=path.read_text()
                assert '训练' in text and '横线' in text
                assert ('标准差' in text if 'brier' in path.name else '95%' in text)
            else:
                assert path.read_bytes().startswith(b'%PDF-')
        ablation_table=Table('data-ablation');ablation_table.feed(evidence_page.read_text())
        assert set(ablation_table.rows)==set(ablations['methods'])
        for variant,method in ablations['methods'].items():
            row=ablation_table.rows[variant]
            assert row[0]==visuals['labels'][variant]
            for index,metric in enumerate(('brier','nll','ece'),start=1):
                value=method['aggregate'][metric]
                assert row[index]==f"{value['mean']:.5f} ± {value['sample_sd']:.5f}"
            value=method['risk_at_30_percent']
            assert row[4]==f"{value['mean']*100:.2f} ± {value['sample_sd']*100:.2f}%"
        report.update(ablation_assets_hashed=6,audited_ablation_rows=3,completed_ablation_runs=6)
        cases=read(ROOT/'site/data/training_ablation_cases_zh.json')
        assert cases['passed'] and cases['test_evaluated'] is False and cases['training_started'] is False
        assert [(c['global_id'],c['candidate_index'],c['radius_cells']) for c in cases['cases']]==[(c['global_id'],c['candidate_index'],c['radius_cells']) for c in data['examples']]
        for path,digest in cases['sources'].items():assert sha(ROOT/path)==digest,path
        assert len(cases['assets'])==12
        for asset in cases['assets']:
            path=ROOT/'site'/asset['path'];assert sha(path)==asset['sha256']
            tree=ET.parse(path);assert not tree.findall('.//{http://www.w3.org/2000/svg}line')
            assert tree.find('{http://www.w3.org/2000/svg}title') is not None
            assert '不画直连路径' in path.read_text()
            assert ('收缩' in path.read_text() if '-footprint' in path.name else '单元格可通行概率' in path.read_text())
        report.update(ablation_case_images=12,fixed_ablation_cases=2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
