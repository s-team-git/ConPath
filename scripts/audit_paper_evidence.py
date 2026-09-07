#!/usr/bin/env python3
"""Read-only checks of manuscript/site evidence, source hashes, and local assets."""

from __future__ import annotations

import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathrel.flatlands_eval import join_flatlands_predictions, _metric_summary


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'img':
            if not attrs.get('alt'):
                raise ValueError('image missing alternative text')
            self.images.append(attrs['src'])


def main():
    page = Page()
    page.feed((ROOT / 'site/index.html').read_text())
    assert len(page.ids) == len(set(page.ids)), 'duplicate HTML ids'
    local_links = 0
    for link in page.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path:
            assert (ROOT / 'site' / unquote(parsed.path)).is_file(), link
        elif parsed.fragment:
            assert parsed.fragment in page.ids, link
        local_links += 1
    json_files = list((ROOT / 'site/data').glob('*.json'))
    for path in json_files:
        json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    source_files = [p for directory in ('src', 'scripts', 'tests') for p in (ROOT / directory).rglob('*.py')]
    for path in source_files:
        ast.parse(path.read_text(), filename=str(path))

    analysis = json.loads((ROOT / 'results/paper_clean_analysis_v1/report.json').read_text())
    assert analysis == json.loads((ROOT / 'site/data/flatlands_clean_paper_analysis.json').read_text())
    software_paths = {
        'script_sha256': 'scripts/analyze_flatlands_clean_validation.py',
        'selective_risk_sha256': 'src/pathrel/selective_risk.py',
        'evaluator_sha256': 'src/pathrel/flatlands_eval.py',
        'paired_comparison_sha256': 'scripts/compare_flatlands_k128_paired.py',
    }
    for key, path in software_paths.items():
        assert analysis['software'][key] == sha(path), path
    predictions = 0
    for method in analysis['methods'].values():
        for seed in method['seeds']:
            path = seed['prediction']
            assert sha(path) == seed['prediction_sha256'], path
            records, _ = join_flatlands_predictions(
                ROOT / path,
                ROOT / 'results/p1_flatlands_query_audit_bounded/selected_observations.csv',
                ROOT / 'results/p1_flatlands_query_audit_bounded/queries.csv', split='validation',
            )
            metrics = _metric_summary(records, weighting='scene', bins=10)
            for key in ('brier', 'nll', 'ece'):
                assert abs(metrics[key] - seed['metrics'][key]) < 1e-12, (path, key)
            assert metrics['count'] == 4224 and metrics['scene_count'] == 142
            predictions += 1
    # This source version is retained because the module later gained a separate
    # mean-map projection helper; the shuffle implementation itself is unchanged.
    historical_helper = subprocess.check_output(
        ['git', 'show', '392017b:src/pathrel/posterior_audits.py'], cwd=ROOT,
    )
    for path in (ROOT / 'results/paper_clean_marginal_shuffle_v1').glob('seed*/run.json'):
        run = json.loads(path.read_text())
        assert run['helper_sha256'] == hashlib.sha256(historical_helper).hexdigest()
        assert run['script_sha256'] == sha('scripts/evaluate_flatlands_marginal_shuffle.py')
        assert all(run['checks'][key] for key in ('canonical_k128_replay_exact', 'empirical_cell_counts_preserved', 'invalid_support_blocked'))

    bounds = json.loads((ROOT / 'site/data/unscenes3d_observation_ceiling.json').read_text())
    assert bounds['software']['script_sha256'] == sha('scripts/audit_unscenes3d_observation_ceiling.py')
    audit = bounds['prediction_bound_audit']
    assert audit['passed'] and audit['event_comparisons'] == 27522
    for value in audit['inputs']:
        assert sha(value['path']) == value['sha256'] and value['bound_violations'] == 0
        assert value['checkpoint_and_hidden_map_replay_exact']
    comparisons = 'results/unscenes3d_ground_valid_support_clamped_k128_comparison_v2/paired_comparison.json'
    assert sha(comparisons) == sha('site/data/unscenes3d_clean_support_k128_candidate.json')
    qualitative = json.loads((ROOT / 'site/data/unscenes3d_clean_candidate_qualitative.json').read_text())
    assert qualitative['forward']['mean_map_projection_version'].startswith('v2:')
    for name, panel in qualitative['panels'].items():
        assert sha(panel['path']) == sha(f'site/assets/unscenes3d_clean_candidate_{name}.png')
    figures = ['flatlands_clean_equal_coverage', 'flatlands_clean_reliability',
               'flatlands_clean_k_convergence', 'flatlands_clean_marginal_shuffle', 'unscenes3d_observation_ceiling']
    for name in figures:
        assert (ROOT / f'site/assets/{name}.pdf').read_bytes().startswith(b'%PDF-')
        assert '<svg' in (ROOT / f'site/assets/{name}.svg').read_text()
    report = {'passed': True, 'test_evaluated': False, 'python_files': len(source_files),
              'site_json_files': len(json_files), 'local_links': local_links, 'image_elements': len(page.images),
              'flatlands_prediction_files_replayed': predictions, 'flatlands_methods': len(analysis['methods']),
              'unscenes3d_bound_comparisons': audit['event_comparisons'], 'standalone_figure_pairs': len(figures)}
    output = ROOT / 'results/maintenance_20260906/paper_evidence_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
