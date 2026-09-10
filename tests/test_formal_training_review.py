"""CPU review regressions for formal schedules, guarded updates and driver receipts."""
import copy
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from pathrel.external_training import flow_accumulated_update, lama_accumulated_update
from pathrel.flow_matching import FlowUNet
from pathrel.formal_training import StagedLRScheduler, checked_flow_update, checked_lama_update
from pathrel.formal_training import (eligible_parent_slots, parent_event_numerator,
                                    parent_weighted_micro_loss)
from pathrel.lama import LaMaBEV, make_discriminator


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('formal_driver_review', ROOT / 'scripts/train_flatlands_formal.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class FormalTrainingReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_models_equal(self, left, right):
        for name, value in left.state_dict().items():
            self.assertTrue(torch.equal(value, right.state_dict()[name]), name)

    def condition(self):
        c = torch.zeros(2, 3, 32, 32)
        c[:, 2, 1:-1, 1:-1] = 1
        c[:, 1, 4:-4, 4:-4] = 1
        c[:, 0, 2, 2] = 1
        return c

    def test_checked_lama_preserves_exact_accumulated_math_and_reports_terms(self):
        torch.manual_seed(72)
        old, old_d = LaMaBEV(ngf=8, n_blocks=1), make_discriminator(ndf=8)
        new, new_d = copy.deepcopy(old), copy.deepcopy(old_d)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=.01)
                      for m in [old, old_d, new, new_d]]
        c = self.condition()
        target = c[:, 2:3].clone()
        expected = lama_accumulated_update(old, old_d, *optimizers[:2], c, target, microbatch=1)
        actual = checked_lama_update(new, new_d, *optimizers[2:], c, target, microbatch=1)
        self.assertEqual(expected, {key: actual[key] for key in expected})
        self.assertIn('adversarial', actual)
        self.assertIn('feature_matching', actual)
        self.assert_models_equal(old, new)
        self.assert_models_equal(old_d, new_d)

    def test_checked_flow_preserves_exact_accumulated_math(self):
        torch.manual_seed(18)
        old = FlowUNet(base_channels=8, heads=2)
        new = copy.deepcopy(old)
        oa = torch.optim.AdamW(old.parameters(), lr=1e-4, weight_decay=.01)
        ob = torch.optim.AdamW(new.parameters(), lr=1e-4, weight_decay=.01)
        c = self.condition()
        target = c[:, 2:3].clone()
        expected = flow_accumulated_update(old, oa, c, target, 1, torch.Generator().manual_seed(14))
        actual = checked_flow_update(new, ob, c, target, 1, torch.Generator().manual_seed(14))
        self.assertEqual(expected, actual)
        self.assert_models_equal(old, new)

    def test_nonfinite_loss_or_gradient_never_reaches_external_optimizer(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.sum()
            @staticmethod
            def backward(ctx, grad):
                return torch.full((1, 1), float('nan'))
        for kind in ('loss', 'gradient'):
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            before = model.weight.detach().clone()
            def invalid_loss(*args, **kwargs):
                return model.weight.sum() * float('nan') if kind == 'loss' else BadGradient.apply(model.weight)
            with mock.patch('pathrel.formal_training.flow_training_loss', invalid_loss), \
                    mock.patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                with self.assertRaises(FloatingPointError):
                    checked_flow_update(model, optimizer, torch.ones(1, 1), torch.ones(1, 1), 1, torch.Generator())
                step.assert_not_called()
            self.assertTrue(torch.equal(before, model.weight))

    def test_full_schedule_can_attach_once_at_pilot_boundary_and_resumes_exactly(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        pilot = StagedLRScheduler(optimizer, pilot_steps=3)
        for _ in range(3):
            pilot.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], .001)
        schedule = StagedLRScheduler(optimizer, pilot_steps=3, full_steps=8)
        schedule.load_state_dict(pilot.state_dict())
        schedule.step()
        state = schedule.state_dict()
        expected = []
        for _ in range(4):
            schedule.step()
            expected.append(optimizer.param_groups[0]['lr'])
        restored_optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=.001)
        restored = StagedLRScheduler(restored_optimizer, pilot_steps=3, full_steps=8)
        restored.load_state_dict(state)
        actual = []
        for _ in range(4):
            restored.step()
            actual.append(restored_optimizer.param_groups[0]['lr'])
        self.assertEqual(expected, actual)
        self.assertAlmostEqual(actual[-1], .0001)
        changed = StagedLRScheduler(restored_optimizer, pilot_steps=3, full_steps=9)
        changed.base_lrs = [.001]
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            changed.load_state_dict(state)

    def test_process_receipts_count_replayed_time_and_mark_killed_session_lower_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            first = driver.ProcessSession(folder, {'step': 0})
            first.heartbeat(last_complete_step=2)
            first.finish(status='failed')
            duration = json.loads((folder / 'sessions' / f'{first.token}.finished.json').read_text())['elapsed_seconds']
            with self.assertRaises(FileExistsError):
                first.finish(status='complete')
            second = driver.ProcessSession(folder, {'step': 0})
            second.heartbeat(last_complete_step=1)
            heartbeat = json.loads((folder / 'sessions' / f'{second.token}.heartbeat.json').read_text())['elapsed_seconds']
            totals = driver.session_totals(folder)
            self.assertAlmostEqual(totals['training_wall_seconds'], duration + heartbeat)
            self.assertTrue(totals['wall_time_is_lower_bound'])
            self.assertEqual(totals['sessions_without_final_receipt'], [second.token])
            second.finish(status='complete')
            self.assertFalse(driver.session_totals(folder)['wall_time_is_lower_bound'])

    def test_source_and_artifact_paths_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(driver, 'ROOT', Path(temporary)):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                driver.project_path('../elsewhere')
            with self.assertRaisesRegex(ValueError, 'omits'):
                driver.verify_sources({'source_hashes': {}})

    def test_existing_boundary_is_checked_without_modifying_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'latest.pt').write_bytes(b'complete-state')
            driver.snapshot(folder, 3)
            digest = hashlib.sha256(b'complete-state').hexdigest()
            receipt = {'completed_steps': 3, 'protocol_sha256': 'protocol', 'checkpoint_sha256': digest}
            (folder / 'boundary_00000003.json').write_text(json.dumps(receipt))
            self.assertTrue(driver.verify_completed_boundary(folder, 3, 'protocol'))
            driver.snapshot(folder, 3)
            self.assertEqual((folder / 'step_00000003.pt').read_bytes(), b'complete-state')
            with self.assertRaises(ValueError):
                driver.verify_completed_boundary(folder, 3, 'changed')

    def test_driver_stop_at_zero_and_just_committed_step_then_idempotent_boundary(self):
        @dataclass
        class Sample:
            row: dict
            starts: np.ndarray
            goals: np.ndarray
            targets: np.ndarray
            candidate_indices: np.ndarray
        samples = [Sample({'parent_group': 'place', 'global_id': f'view{i}'}, np.array([[1, 1]]),
                          np.array([[2, 2]]), np.zeros((1, 3)), np.array([0])) for i in range(2)]
        original_generator = torch.Generator
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            protocol = {'source_hashes': {}, 'seeds': [11], 'methods': {'flow': {'members': 1,
                'batch_size': 1, 'learning_rate': .001, 'weight_decay': .01, 'betas': [.9, .999]}},
                'output_root': 'results', 'staged': {'smoke_steps': 1, 'pilot_steps': 3},
                'training_runtime': {'checkpoint_interval': 1, 'cpu_threads': 1, 'allocator_limit_gib': 28},
                'member_seeds': {'flow': {'11': [11]}}, 'data_root': 'data',
                'query': {'training_queries_per_observation': 1}, 'partition': {'train': 1, 'train_views_per_parent': 2}}
            (root / 'data').mkdir()
            (root / 'data/seal.json').write_text('{"fixture": true}\n')
            protocol['data_seal_sha256'] = hashlib.sha256((root / 'data/seal.json').read_bytes()).hexdigest()
            path = root / 'protocol.json'
            path.write_text(json.dumps(protocol))
            (root / 'results').mkdir()
            stop_path = root / 'results/STOP'
            stop_path.touch()
            stack.enter_context(mock.patch.object(driver, 'ROOT', root))
            stack.enter_context(mock.patch.object(driver, 'verify_sources'))
            stack.enter_context(mock.patch.object(driver, 'configure_reproducible_cuda'))
            stack.enter_context(mock.patch.object(driver.signal, 'signal'))
            loader = stack.enter_context(mock.patch('pathrel.formal_data.load_formal_data', return_value=samples))
            factory = stack.enter_context(mock.patch('pathrel.formal_inference.make_models',
                side_effect=lambda *args, **kwargs: {'model': torch.nn.Linear(1, 1)}))
            stack.enter_context(mock.patch.object(torch, 'Generator', side_effect=lambda **kw: original_generator(device='cpu')))
            for name in ('set_per_process_memory_fraction', 'reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(mock.patch.object(torch.cuda, name))
            stack.enter_context(mock.patch.object(torch.cuda, 'get_device_properties',
                return_value=types.SimpleNamespace(total_memory=100 * 1024 ** 3)))
            base_args = ['train_flatlands_formal.py', '--protocol', str(path), '--method', 'flow',
                         '--seed', '11', '--stop-step', '3']
            folder = root / 'results/runs/flow/11/member_0'
            driver.STOP = False
            with mock.patch.object(driver.sys, 'argv', base_args), self.assertRaises(SystemExit) as stop_zero:
                driver.main()
            self.assertEqual(stop_zero.exception.code, 130)
            self.assertEqual(torch.load(folder / 'latest.pt', weights_only=False)['progress_index'], 0)
            stop_path.unlink()
            def update_and_pause(*args, **kwargs):
                driver.STOP = True
                return {'velocity_mse': .5}
            with mock.patch.object(driver.sys, 'argv', base_args + ['--resume']), \
                    mock.patch.object(driver, 'training_update', side_effect=update_and_pause), \
                    self.assertRaises(SystemExit) as stop_committed:
                driver.main()
            self.assertEqual(stop_committed.exception.code, 130)
            self.assertEqual(torch.load(folder / 'latest.pt', weights_only=False)['progress_index'], 1)
            driver.STOP = False
            with mock.patch.object(driver.sys, 'argv', base_args + ['--resume']), \
                    mock.patch.object(driver, 'training_update', return_value={'velocity_mse': .4}):
                driver.main()
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')}
            calls = factory.call_count
            with mock.patch.object(driver.sys, 'argv', base_args + ['--resume']):
                driver.main()
            self.assertEqual(calls, factory.call_count)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')}
            self.assertEqual(before, after)
            self.assertEqual(len(list((folder / 'sessions').glob('*.finished.json'))), 3)
            self.assertFalse(json.loads((folder / 'time_summary.json').read_text())['wall_time_is_lower_bound'])
            driver.STOP = False
            calls_before = loader.call_count
            protocol['output_root'] = 'bad_seal_run'
            protocol['data_seal_sha256'] = 'wrong'
            path.write_text(json.dumps(protocol))
            with mock.patch.object(driver.sys, 'argv', base_args), self.assertRaisesRegex(ValueError, 'data seal'):
                driver.main()
            self.assertEqual(loader.call_count, calls_before)
            protocol['output_root'] = 'bad_parent_count_run'
            protocol['data_seal_sha256'] = hashlib.sha256((root / 'data/seal.json').read_bytes()).hexdigest()
            protocol['partition']['train'] = 2
            path.write_text(json.dumps(protocol))
            calls_before = factory.call_count
            with mock.patch.object(driver.sys, 'argv', base_args), self.assertRaisesRegex(ValueError, 'parent count'):
                driver.main()
            self.assertEqual(factory.call_count, calls_before)

    def test_one_and_two_query_view_parents_have_equal_expected_loss_and_gradient(self):
        config = {'_views_per_parent': 2, '_n_query_views_by_parent': {'one': 1, 'two': 2, 'none': 0}}
        def sample(parent, active):
            return types.SimpleNamespace(row={'parent_group': parent}, starts=[0] if active else [])
        choices = [(sample('one', True), 3.), (sample('one', False), 0.)]
        other = [(sample('two', True), 5.), (sample('two', True), 9.)]
        value, gradient = [], []
        for a, a_loss in choices:
            for b, b_loss in other:
                parameter = torch.tensor(2., dtype=torch.float64, requires_grad=True)
                samples = [a, b, sample('none', False)]
                losses = parameter * torch.tensor([a_loss, b_loss, 0.], dtype=torch.float64)
                numerator = parent_event_numerator(losses, samples, config)
                self.assertEqual(eligible_parent_slots(samples, config), 2)
                loss = numerator / eligible_parent_slots(samples, config)
                loss.backward()
                value.append(float(loss.detach())); gradient.append(float(parameter.grad))
        # Parent one contributes its only valid-view loss (3); parent two
        # contributes the mean over two valid views (7). Both parent weights 1/2.
        self.assertEqual(sum(value) / 4, 10.)
        self.assertEqual(sum(gradient) / 4, 5.)

    def test_parent_event_microbatch_accumulation_invariant_including_empty_view_slots(self):
        samples = [types.SimpleNamespace(row={'parent_group': parent}, starts=[0] if active else [])
                   for parent, active in [('one', False), ('two', True), ('one', True), ('none', False)]]
        config = {'_views_per_parent': 2, '_n_query_views_by_parent': {'one': 1, 'two': 2, 'none': 0}}
        eligible = eligible_parent_slots(samples, config)
        self.assertEqual(eligible, 3)  # Both draws of parent one count, even its empty view.
        answers = []
        for micro in (1, 2, 3, 4):
            parameter = torch.tensor(2., dtype=torch.float64, requires_grad=True)
            total = 0.
            for begin in range(0, 4, micro):
                end = min(begin + micro, 4)
                view_losses = parameter * torch.tensor([0., 4., 3., 0.], dtype=torch.float64)[begin:end]
                numerator = parent_event_numerator(view_losses, samples[begin:end], config)
                map_loss = parameter * torch.arange(1., 5., dtype=torch.float64)[begin:end].mean()
                loss = parent_weighted_micro_loss(map_loss, parameter * .2, numerator,
                    micro_count=end - begin, batch_count=4, batch_eligible_parents=eligible,
                    event_weight=2., variogram_weight=.1)
                loss.backward()
                total += float(loss.detach())
            answers.append((total, float(parameter.grad)))
        for value in answers[1:]:
            self.assertAlmostEqual(value[0], answers[0][0], places=13)
            self.assertAlmostEqual(value[1], answers[0][1], places=13)

    def test_full_plan_requires_unchanged_numeric_visual_and_pilot_evaluation_receipts(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(driver, 'ROOT', Path(temporary)):
            root = Path(temporary)
            protocol = {'staged': {'smoke_steps': 1000, 'pilot_steps': 5000, 'seed': 20260831,
                                   'allowed_full_steps': [10000, 20000]}}
            identity = {'protocol_sha256': 'protocol', 'method': 'flow'}
            plan = {**identity, 'pilot_gate_passed': True, 'full_steps': 10000,
                    'checkpoint_steps': [1000, 5000, 7500, 10000]}
            with self.assertRaisesRegex(ValueError, 'requires'):
                driver.verify_full_plan(plan, protocol, 'protocol', 'flow')
            metrics = {**identity, 'outer_seed': 20260831, 'stage': 'pilot'}
            (root / 'metrics.json').write_text(json.dumps(metrics))
            numeric = {**identity, 'passed': True, 'seed': 20260831,
                       'pilot_evaluation_metrics': 'metrics.json',
                       'pilot_evaluation_metrics_sha256': driver.sha(root / 'metrics.json')}
            visual = {**identity, 'passed': True, 'seed': 20260831}
            for name, row in [('numeric', numeric), ('visual', visual)]:
                (root / f'{name}.json').write_text(json.dumps(row))
            plan.update(pilot_gate_receipt='numeric.json', pilot_gate_sha256=driver.sha(root / 'numeric.json'),
                        visual_review_receipt='visual.json', visual_review_sha256=driver.sha(root / 'visual.json'))
            self.assertEqual(driver.verify_full_plan(plan, protocol, 'protocol', 'flow'), 10000)
            (root / 'visual.json').write_text(json.dumps({**visual, 'passed': False}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                driver.verify_full_plan(plan, protocol, 'protocol', 'flow')
            plan['visual_review_sha256'] = driver.sha(root / 'visual.json')
            with self.assertRaisesRegex(ValueError, 'pass status'):
                driver.verify_full_plan(plan, protocol, 'protocol', 'flow')
            (root / 'visual.json').write_text(json.dumps(visual))
            plan['visual_review_sha256'] = driver.sha(root / 'visual.json')
            (root / 'metrics.json').write_text(json.dumps({**metrics, 'outer_seed': 20260901}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                driver.verify_full_plan(plan, protocol, 'protocol', 'flow')
            numeric['pilot_evaluation_metrics_sha256'] = driver.sha(root / 'metrics.json')
            (root / 'numeric.json').write_text(json.dumps(numeric))
            plan['pilot_gate_sha256'] = driver.sha(root / 'numeric.json')
            with self.assertRaisesRegex(ValueError, 'different pilot'):
                driver.verify_full_plan(plan, protocol, 'protocol', 'flow')
            plan['pilot_gate_receipt'] = '../outside.json'
            with self.assertRaisesRegex(ValueError, 'escapes'):
                driver.verify_full_plan(plan, protocol, 'protocol', 'flow')

    def test_full_plan_rejects_unregistered_budget_or_checkpoint_schedule(self):
        protocol = {'staged': {'smoke_steps': 1000, 'pilot_steps': 5000, 'seed': 20260831,
                               'allowed_full_steps': [10000, 20000]}}
        plan = {'method': 'lama', 'protocol_sha256': 'protocol', 'pilot_gate_passed': True,
                'full_steps': 300000, 'checkpoint_steps': [1000, 5000, 300000]}
        with self.assertRaisesRegex(ValueError, 'outside'):
            driver.verify_full_plan(plan, protocol, 'protocol', 'lama')
        plan['full_steps'] = 10000
        plan['checkpoint_steps'] = [1000, 5000, 10000]
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            driver.verify_full_plan(plan, protocol, 'protocol', 'lama')
        with self.assertRaisesRegex(ValueError, 'method/protocol'):
            driver.verify_full_plan(plan, protocol, 'protocol', 'flow')



if __name__ == '__main__':
    unittest.main()
