"""Mutation tests for the publication gate's seed, order and uncertainty checks."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.audit_coherent_results import (
    MAP_METRICS, METHODS, SEEDS, audit_aggregate, recompute_selection,
)


def fixture():
    samples = [SimpleNamespace(row={'parent_group': f'p{i:02}', 'source_dataset': f's{i // 8}'},
                               targets=np.zeros((13 if i < 39 else 8, 3), dtype=bool)) for i in range(40)]
    parents = [s.row['parent_group'] for s in samples]
    sources = [s.row['source_dataset'] for s in samples]
    reports, methods, paired = {}, {}, {}
    for mi, method in enumerate(METHODS):
        k = 1 if method == 'deterministic' else 32
        brier = [.10 + mi * .02 + si * .005 for si in range(2)]
        risk = [.20 + mi * .02 + si * .01 for si in range(2)]
        maps = {name: None if name == 'masked_energy_score' and k == 1 else .4 for name in MAP_METRICS}
        for si, seed in enumerate(SEEDS):
            reports[method, seed] = {'budgets': {str(k): {'parents': parents.copy(), 'sources': sources.copy(),
                                      'events': 1545, 'brier': brier[si], 'risk30': risk[si],
                                      'parent_brier': [brier[si]] * 40, 'map': maps.copy()}}}
        methods[method] = {'seeds': list(SEEDS), 'samples': k, 'map': maps.copy()}
        for name, values in [('brier', brier), ('risk30', risk)]:
            methods[method][name] = {'values': values, 'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}
        if mi:
            paired[method] = {'other_minus_conpath': mi * .02, 'ci95': [mi * .02, mi * .02],
                              'unit': 'parent place, source stratified; conditional on the displayed two-seed average'}
    analysis = {'methods': methods, 'paired_brier': paired,
                'screen_conditions': {'brier_better_than_matched_conpath': True, 'risk30_not_worse_than_matched_conpath': True},
                'development_screen_passed': True, 'training_seeds': list(SEEDS),
                'validation_parents': 40, 'validation_events': 1545,
                'validation_reused_for_model_development': True, 'final_test': False,
                'no_direct_published_paper_ranking': True, 'new_physical_test_reads': 0}
    return analysis, reports, samples


class CoherentAggregateAuditTests(unittest.TestCase):
    def test_valid_two_seed_fixture(self):
        analysis, reports, samples = fixture()
        self.assertEqual(audit_aggregate(analysis, reports, samples)['method_seed_reports'], 8)

    def test_rejects_three_seed_comparator(self):
        analysis, reports, samples = fixture()
        analysis['methods']['correlated']['seeds'].append(20260912)
        with self.assertRaises(AssertionError):
            audit_aggregate(analysis, reports, samples)

    def test_rejects_population_sd(self):
        analysis, reports, samples = fixture()
        score = analysis['methods']['coherent_categorical']['brier']
        score['sd'] = float(np.std(score['values'], ddof=0))
        with self.assertRaises(AssertionError):
            audit_aggregate(analysis, reports, samples)

    def test_rejects_parent_permutation(self):
        analysis, reports, samples = fixture()
        reports['correlated', SEEDS[0]]['budgets']['32']['parents'].reverse()
        with self.assertRaises(AssertionError):
            audit_aggregate(analysis, reports, samples)

    def test_rejects_truthy_non_boolean_screen(self):
        for value in ('true', 1):
            analysis, reports, samples = fixture()
            analysis['development_screen_passed'] = value
            with self.assertRaises(AssertionError):
                audit_aggregate(analysis, reports, samples)

    def test_rejects_interval_drift_and_seed_order(self):
        for field in ('interval', 'seed_order'):
            analysis, reports, samples = fixture()
            if field == 'interval':
                analysis['paired_brier']['correlated']['ci95'][0] -= .01
            else:
                analysis['methods']['correlated']['brier']['values'].reverse()
            with self.assertRaises(AssertionError):
                audit_aggregate(analysis, reports, samples)

    def test_rejects_final_test_claim(self):
        analysis, reports, samples = fixture()
        analysis['final_test'] = True
        with self.assertRaises(AssertionError):
            audit_aggregate(analysis, reports, samples)


class SelectionAuditTests(unittest.TestCase):
    def test_map_tiebreak_can_select_slightly_higher_brier(self):
        config = {'minimum_epochs': 1, 'max_epochs': 3, 'patience': 2}
        history = [
            {'epoch': 1, 'calibration': {'event_brier': .2, 'map_nll': .5}, 'improved': True, 'best_epoch': 1, 'best_brier': .2, 'patience': 0},
            {'epoch': 2, 'calibration': {'event_brier': .200001, 'map_nll': .49}, 'improved': True, 'best_epoch': 2, 'best_brier': .200001, 'patience': 0},
            {'epoch': 3, 'calibration': {'event_brier': .21, 'map_nll': .48}, 'improved': False, 'best_epoch': 2, 'best_brier': .200001, 'patience': 1},
        ]
        self.assertEqual(recompute_selection(history, config), ((.200001, .49), 2, 1))
        invalid = deepcopy(history)
        invalid[1]['best_epoch'] = 1
        with self.assertRaises(AssertionError):
            recompute_selection(invalid, config)


if __name__ == '__main__':
    unittest.main()
