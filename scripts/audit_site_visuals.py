#!/usr/bin/env python3
"""Independently check exported media provenance and the Chinese result table."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


class Table(HTMLParser):
    def __init__(self):super().__init__();self.method=None;self.rows={};self.in_cell=False;self.cells=[];self.text=''
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='tr' and 'data-method' in attrs:self.method=attrs['data-method'];self.cells=[]
        if self.method and tag in ('td','th'):self.in_cell=True;self.text=''
    def handle_data(self,data):
        if self.in_cell:self.text+=data
    def handle_endtag(self,tag):
        if self.in_cell and tag in ('td','th'):self.cells.append(self.text);self.in_cell=False
        if self.method and tag=='tr':self.rows[self.method]=self.cells;self.method=None


def main():
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
    table=Table();table.feed((ROOT/'site/index.html').read_text())
    methods=read(ROOT/'site/data/flatlands_clean_paper_analysis.json')['methods']
    assert set(table.rows)==set(methods)
    for key,method in methods.items():
        row=table.rows[key];assert int(row[1])==method['seed_count']
        for index,metric in [(2,'brier'),(3,'nll'),(4,'ece')]:
            values=row[index].split(' ± ');expected=method['aggregate'][metric]
            assert values[0]==f"{expected['mean']:.5f}"
            if expected['sample_sd'] is not None:assert values[1]==f"{expected['sample_sd']:.5f}"
        assert row[5]==f"{method['equal_coverage']['0.3']['mean']*100:.2f}%"
    pause=read(ROOT/'results/paper_clean_ablation_matrix_v1/progress.json')
    assert pause['pause']['automatic_resume_allowed'] is False
    report={'passed':True,'test_evaluated':False,'training_resumed':False,'assets_hashed':len(data['assets']),'unchanged_camera_images':len(photos),'gallery_scenes':12,'training_sequence_frames':18,'fixed_validation_examples':2,'audited_method_rows':len(table.rows),'source_hashes':len(data['sources'])}
    (ROOT/'results/site_redesign_20260907/visual_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
