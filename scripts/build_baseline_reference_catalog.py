#!/usr/bin/env python3
"""Extract identified published table entries without ranking unmatched experiments."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path('results/baseline_review_20260909_v1')
    text_path = Path('results/literature_protocol_20260907/flatlands.txt')
    paper = text_path.read_text()
    rows = []
    sections = [('3', 'Table 3: Fidelity metrics', 'Table 4:', ['UMR', 'IoU', 'F1']),
                ('4', 'Table 4: Stochastic evaluation', 'Masked Energy Score. For multi-sample', ['MES', 'mean_of_K_IoU', 'pixel_variance'])]
    stochastic = {'LaMa-Ens.', 'Diffusion', 'Flow Match.', 'FM+XAttn'}
    for table, start, end, metrics in sections:
        block = paper[paper.index(start):]; block = block[:block.index(end)]
        count = 0
        for line in block.splitlines():
            values = re.findall(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
            if len(values) != 6: continue
            name = re.split(r'\d+\.\d+', line, maxsplit=1)[0].strip()
            count += 1
            for j, (mean, std) in enumerate(values):
                selection = 'oracle_best_of_K' if table == '3' and name in stochastic else 'set_statistic' if table == '4' else 'single_output'
                rows.append({'paper': 'FlatLands', 'version': '2603.16016v3', 'table': table, 'method': name,
                             'split': 'ID' if j < 3 else 'OOD_ScanNetPP', 'samples': 4 if name in stochastic else 1,
                             'metric': metrics[j%3], 'mean': float(mean), 'reported_std': float(std),
                             'sample_selection': selection, 'direct_conpath_ranking_allowed': False,
                             'source_url': 'https://arxiv.org/html/2603.16016v3',
                             'dispersion_scope': 'Paper-reported mean/std; not assumed to be3 independent training repeats.'})
        assert count == (11 if table == '3' else 4), (table, count)
    repo = json.loads((root/'flatlands_repo.json').read_text())
    hub = json.loads((root/'flatlands_hf.json').read_text())
    archive = next(r for r in hub if r['path'] == 'FlatLands_final_dataset.zip')
    assert archive['lfs']['oid'] == 'e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f'
    candidate_models = [r['path'] for r in repo['tree'] if r['path'].endswith(('.pt', '.pth', '.ckpt', '.py'))]
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'published_reference_only': True,
              'direct_conpath_ranking_allowed': False, 'rows': rows,
              'compatibility_failures': ['Our primary score is path-event Brier; paper MES is a map-set score and IoU is overlap.',
                 'Our160-observation provenance validation is not the canonical12129 ID/16214 OOD test.',
                 'Official physical splits overlap subscene IDs; our old subscene-disjoint protocol additionally has parent-place overlap in27/160 validation observations.',
                 'Our learned models used160 training observations; paper reports215342 training observations.',
                 'Oracle best-of4 cannot be compared to first sample or event average. Training-seed SD and reported observation dispersion differ.',
                 'Physical scale and exact author evaluator are not independently resolved.'],
              'release_audit': {'repository_revision': repo['sha'], 'model_or_python_files': candidate_models,
                                'hub_archive_sha256': archive['lfs']['oid'], 'archive_changed_since_local_audit': False,
                                'weights_and_predictions_available_in_checked_locations': bool(candidate_models)},
              'attention_ablation_reported': {'source_table':'S5','setting':'OOD K4 CFG2','full_MES':.095,'without_cross_attention_MES':.096,
                                              'interpretation':'Small reported change; does not establish that adding attention will fix our outdoor observation model.'},
              'source_hashes': {str(p):sha(p) for p in [text_path, Path('results/literature_protocol_20260907/flatlands.pdf'), root/'flatlands_repo.json', root/'flatlands_hf.json', Path(__file__)]}}
    path=Path('site/data/published_baseline_reference.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with Path('site/data/published_baseline_reference.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (root/'reference_catalog_receipt.json').write_text(json.dumps({'entries':len(rows),'json_sha256':sha(path),'csv_sha256':sha(Path('site/data/published_baseline_reference.csv'))},indent=2)+'\n')
    print(json.dumps({'entries':len(rows),'repository_revision':repo['sha'],'model_files':candidate_models}))


if __name__ == '__main__':main()
