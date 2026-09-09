"""Synthetic report corruption tests; never load cached packets or models."""
import copy
import statistics
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.audit_parent_pilot_results import (
    audit_analysis, audit_parent_summary, MAP_METRICS, RULE_METHODS, SEEDS, TRAINED_METHODS,
)
from scripts.evaluate_parent_group_pilot import paired_interval


def synthetic_reports():
    samples = [SimpleNamespace(row={'parent_group': f'parent{i}', 'source_dataset': source},
                               targets=np.zeros((i+1, 3), dtype=bool))
               for i, source in enumerate(('a', 'a', 'b', 'b', 'b'))]
    reports = []
    for mi, method in enumerate((*TRAINED_METHODS, *RULE_METHODS)):
        for si, seed in enumerate(SEEDS if method in TRAINED_METHODS else ('rule',)):
            budgets = {}
            for k in (('4', '32') if method in ('correlated', 'independent') else ('1',)):
                # Unequal seed shifts and distinct K4/K32 scores expose ordering,
                # ddof, and accidentally selecting the small-budget result.
                base = .08 + .035*mi + (0., .011, .036)[si] + (.2 if k == '4' else 0.)
                vector = [base + i*(.004 + mi*.002) for i in range(len(samples))]
                budgets[k] = {
                    'brier': statistics.mean(vector), 'parent_brier': vector,
                    'risk30': .12 + mi*.025 + (0., .019, .041)[si],
                    'parents': [s.row['parent_group'] for s in samples],
                    'sources': [s.row['source_dataset'] for s in samples],
                    'events': sum(s.targets.size for s in samples),
                    'by_source': {name: statistics.mean(v for v, s in zip(vector, samples)
                                                       if s.row['source_dataset'] == name)
                                  for name in ('a', 'b')},
                    'map': None if method == 'train_radius_prior' else {
                        name: None if name == 'masked_energy_score' and k == '1' else base + j*.01
                        for j, name in enumerate(MAP_METRICS)},
                }
            reports.append({'method': method, 'seed': seed, 'budgets': budgets})
    return samples, reports


def synthetic_analysis(samples, reports):
    """Construct fixture summaries with Python statistics, not audit helpers."""
    analysis = {'methods': {}, 'paired_brier': {}, 'training_runs_completed': 9,
                'validation_parents': len(samples), 'validation_events': sum(s.targets.size for s in samples)}
    parent_means, primary = {}, {}
    for method in (*TRAINED_METHODS, *RULE_METHODS):
        group = [r for r in reports if r['method'] == method]
        combined = {'repeats': len(group), 'budgets': {}}
        for k in group[0]['budgets']:
            summaries = [r['budgets'][k] for r in group]
            aggregate = {}
            for metric in ('brier', 'risk30'):
                values = [s[metric] for s in summaries]
                aggregate[metric] = {'values': values, 'mean': statistics.mean(values),
                                     'sd': statistics.stdev(values) if len(values) > 1 else None}
            aggregate['map'] = None if method == 'train_radius_prior' else {
                metric: None if summaries[0]['map'][metric] is None else statistics.mean(s['map'][metric] for s in summaries)
                for metric in MAP_METRICS}
            combined['budgets'][k] = aggregate
        analysis['methods'][method] = combined
        k = '32' if method in ('correlated', 'independent') else '1'
        parent_means[method] = [statistics.mean(v) for v in zip(*(r['budgets'][k]['parent_brier'] for r in group))]
        primary[method] = combined['budgets'][k]
    rng = np.random.default_rng(20260909)
    sources = [s.row['source_dataset'] for s in samples]
    for method in (*TRAINED_METHODS[1:], *RULE_METHODS):
        analysis['paired_brier'][method] = paired_interval(parent_means['correlated'], parent_means[method], sources, rng)
    conditions = {f'brier_better_than_{m}': primary['correlated']['brier']['mean'] < primary[m]['brier']['mean']
                  for m in ('independent', 'deterministic', 'all_floor')}
    conditions['risk30_not_worse_than_deterministic'] = primary['correlated']['risk30']['mean'] <= primary['deterministic']['risk30']['mean']
    analysis['screen_conditions'] = conditions
    analysis['research_screen_passed'] = all(conditions.values())
    return analysis


class ParentPilotResultsAuditTests(unittest.TestCase):
    def setUp(self):
        self.samples, self.reports = synthetic_reports()
        self.analysis = synthetic_analysis(self.samples, self.reports)

    def test_complete_valid_report_set_in_any_file_order(self):
        result = audit_analysis(self.analysis, list(reversed(self.reports)), self.samples)
        self.assertTrue(result['passed'])
        self.assertEqual(result['method_seed_reports'], 13)
        self.assertEqual(result['method_budget_aggregates'], 9)
        self.assertTrue(result['recomputed_research_screen_passed'])

    def test_csv_identity_join_detects_swapped_parent_scores_with_same_mean(self):
        summary = self.reports[0]['budgets']['32']
        errors = {s.row['parent_group']: [v]*s.targets.size
                  for s, v in zip(self.samples, summary['parent_brier'])}
        audit_parent_summary(summary, dict(reversed(list(errors.items()))), self.samples)
        broken = copy.deepcopy(summary)
        broken['parent_brier'][0], broken['parent_brier'][1] = broken['parent_brier'][1], broken['parent_brier'][0]
        with self.assertRaisesRegex(AssertionError, 'CSV parent Brier'):
            audit_parent_summary(broken, errors, self.samples)
        for field, value in [('parents', list(reversed(summary['parents']))),
                             ('sources', list(reversed(summary['sources']))),
                             ('events', summary['events']+1), ('by_source', {'a': 0., 'b': 0.})]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit_parent_summary(dict(summary, **{field: value}), errors, self.samples)

    def test_rejects_missing_duplicate_and_wrong_seed_reports(self):
        variants = [self.reports[:-1], self.reports[:-1]+[self.reports[0]],
                    [dict(self.reports[0], seed=1), *self.reports[1:]]]
        for reports in variants:
            with self.subTest(count=len(reports)), self.assertRaises(AssertionError):
                audit_analysis(self.analysis, reports, self.samples)

    def test_rejects_corrupted_seed_values_mean_sample_sd_and_repeats(self):
        for field in ('values', 'mean', 'sd', 'repeats'):
            broken = copy.deepcopy(self.analysis)
            method = broken['methods']['correlated']
            metric = method['budgets']['32']['brier']
            if field == 'values':
                metric[field].reverse()
            elif field == 'repeats':
                method[field] = 2
            elif field == 'sd':
                metric[field] = statistics.pstdev(metric['values'])
            else:
                metric[field] += .01
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit_analysis(broken, self.reports, self.samples)

    def test_rejects_wrong_budget_map_mean_and_null_conventions(self):
        for case in ('budget', 'map_mean', 'k1_mes', 'prior_map', 'rule_sd'):
            broken = copy.deepcopy(self.analysis)
            if case == 'budget':
                budgets = broken['methods']['correlated']['budgets']
                budgets['32'] = copy.deepcopy(budgets['4'])
            elif case == 'map_mean':
                broken['methods']['correlated']['budgets']['32']['map']['mean_iou'] += .01
            elif case == 'k1_mes':
                broken['methods']['deterministic']['budgets']['1']['map']['masked_energy_score'] = 0.
            elif case == 'prior_map':
                broken['methods']['train_radius_prior']['budgets']['1']['map'] = {}
            else:
                broken['methods']['all_floor']['budgets']['1']['brier']['sd'] = 0.
            with self.subTest(case=case), self.assertRaises(AssertionError):
                audit_analysis(broken, self.reports, self.samples)

    def test_rejects_parent_or_source_order_and_zero_query_parent(self):
        for field in ('parents', 'sources', 'parent_brier'):
            reports = copy.deepcopy(self.reports)
            reports[0]['budgets']['32'][field].reverse()
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit_analysis(self.analysis, reports, self.samples)
        samples = copy.deepcopy(self.samples)
        samples[0].targets = np.empty((0, 3), dtype=bool)
        with self.assertRaisesRegex(AssertionError, 'no queries'):
            audit_analysis(self.analysis, self.reports, samples)

    def test_rejects_wrong_paired_sign_interval_and_unit(self):
        for field in ('other_minus_conpath', 'ci95', 'unit'):
            broken = copy.deepcopy(self.analysis)
            paired = broken['paired_brier']['independent']
            if field == 'other_minus_conpath':
                paired[field] *= -1
            elif field == 'ci95':
                paired[field] = [paired['other_minus_conpath']]*2
            else:
                paired[field] = 'independent events'
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit_analysis(broken, self.reports, self.samples)

    def test_rejects_screen_tampering_missing_fields_and_integer_booleans(self):
        for case in ('condition', 'decision', 'missing', 'extra', 'integer'):
            broken = copy.deepcopy(self.analysis)
            if case == 'decision':
                broken['research_screen_passed'] = False
            elif case == 'missing':
                broken['screen_conditions'].pop('brier_better_than_independent')
            elif case == 'extra':
                broken['screen_conditions']['extra'] = True
            else:
                broken['screen_conditions']['brier_better_than_independent'] = 1 if case == 'integer' else False
            with self.subTest(case=case), self.assertRaises(AssertionError):
                audit_analysis(broken, self.reports, self.samples)

    def test_each_screen_failure_and_boundary_uses_primary_budget(self):
        for failing_method in ('independent', 'deterministic', 'all_floor', 'risk'):
            reports = copy.deepcopy(self.reports)
            for report in reports:
                k = '32' if report['method'] in ('correlated', 'independent') else '1'
                if report['method'] == failing_method:
                    for summary in report['budgets'].values():
                        summary['parent_brier'] = [0.]*len(self.samples)
                        summary['brier'] = 0.
                if failing_method == 'risk' and report['method'] == 'deterministic':
                    report['budgets'][k]['risk30'] = 0.
            analysis = synthetic_analysis(self.samples, reports)
            self.assertFalse(analysis['research_screen_passed'])
            audit_analysis(analysis, reports, self.samples)
        reports = copy.deepcopy(self.reports)
        for r in reports:
            if r['method'] == 'deterministic':
                r['budgets']['1']['risk30'] = .2
            if r['method'] == 'correlated':
                r['budgets']['32']['risk30'] = .2
            if r['method'] == 'independent':
                source = next(c for c in reports if c['method'] == 'correlated' and c['seed'] == r['seed'])
                r['budgets']['32']['parent_brier'] = source['budgets']['32']['parent_brier'][:]
                r['budgets']['32']['brier'] = source['budgets']['32']['brier']
        analysis = synthetic_analysis(self.samples, reports)
        self.assertFalse(analysis['screen_conditions']['brier_better_than_independent'])
        self.assertTrue(analysis['screen_conditions']['risk30_not_worse_than_deterministic'])
        audit_analysis(analysis, reports, self.samples)

    def test_rejects_nonfinite_report_and_analysis_numbers(self):
        for bad in (float('nan'), float('inf')):
            broken = copy.deepcopy(self.analysis)
            broken['methods']['correlated']['budgets']['32']['brier']['mean'] = bad
            with self.subTest(value=bad), self.assertRaises(AssertionError):
                audit_analysis(broken, self.reports, self.samples)
            reports = copy.deepcopy(self.reports)
            reports[0]['budgets']['32']['parent_brier'][0] = bad
            with self.assertRaises(AssertionError):
                audit_analysis(self.analysis, reports, self.samples)


if __name__ == '__main__':
    unittest.main()
