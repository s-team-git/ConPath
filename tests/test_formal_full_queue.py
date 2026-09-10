"""CPU-only controller proofs with mocked children and synthetic artifact receipts."""
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('formal_full_queue_test', ROOT / 'scripts/run_flatlands_formal_full.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class Fixture:
    """Small 1/5/7/10-step analog; actual frozen budgets are not executed."""
    def __init__(self, root, method='flow'):
        self.root, self.method = root, method
        self.out = root / 'out'
        self.out.mkdir()
        members = 4 if method == 'lama' else 1
        self.protocol = {'seeds': queue.SEEDS, 'staged': {'seed': queue.SEEDS[0], 'smoke_steps': 1,
            'pilot_steps': 5, 'allowed_full_steps': [10, 20]}, 'output_root': 'out', 'data_root': 'data',
            'data_seal_sha256': 'seal', 'methods': {method: {'members': members}},
            'member_seeds': {method: {str(seed): [seed + m * 100 for m in range(members)] for seed in queue.SEEDS}},
            'scenes': {'calibration': [{'global_id': 'cal'}], 'validation': [{'global_id': 'val'}]},
            'test_lock': {'final_test_locked': True, 'location_6_locked': True}}
        self.protocol_path = root / 'protocol.json'
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.digest = queue.sha256(self.protocol_path)
        self.plan = {'full_steps': 10, 'checkpoint_steps': [1, 5, 7, 10]}
        path = self.out / 'full_plans' / f'{method}.json'
        path.parent.mkdir()
        path.write_text(json.dumps(self.plan))
        self.info = {'plan': self.plan, 'path': path, 'sha256': queue.sha256(path)}
        self.calls = []
        self.fail_once = False

    def train(self, seed, member, step):
        folder = self.out / 'runs' / self.method / str(seed) / f'member_{member}'
        folder.mkdir(parents=True, exist_ok=True)
        prior_steps = len((folder / 'progress.jsonl').read_text().splitlines()) if (folder / 'progress.jsonl').exists() else 0
        checkpoint = folder / f'step_{step:08d}.pt'
        checkpoint.write_bytes(f'{self.method}/{seed}/{member}/{step}'.encode())
        queue.atomic_json(folder / f'boundary_{step:08d}.json', {'method': self.method, 'seed': seed,
            'member': member, 'completed_steps': step, 'protocol_sha256': self.digest,
            'checkpoint_sha256': queue.sha256(checkpoint)})
        rows = [{'step': i, 'losses': {'loss': .1}, 'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200}
                for i in range(1, step + 1)]
        (folder / 'progress.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        sessions = folder / 'sessions'
        sessions.mkdir(exist_ok=True)
        token = f'step_{step:08d}'
        queue.atomic_json(sessions / (token + '.started.json'), {'session_id': token})
        queue.atomic_json(sessions / (token + '.finished.json'), {'session_id': token,
            'elapsed_seconds': (step - prior_steps) / 10})
        queue.atomic_json(folder / 'time_summary.json', queue.session_totals(folder))
        return checkpoint

    def score(self, step):
        # A short checkpoint actually wins. Its result MUST NOT be replaced by
        # an inferior sufficiently long checkpoint for publication eligibility.
        return {1: .12, 5: .1, 7: .101, 10: .1015}[step]

    def emit_evaluation(self, output, seed, step, stage, checkpoints, splits, calibration_file=None):
        output.mkdir(parents=True, exist_ok=False)
        provenance = [{'member_index': member, 'initialization_seed': self.protocol['member_seeds'][self.method][str(seed)][member],
            'checkpoint': queue.relative(path), 'checkpoint_sha256': queue.sha256(path), 'completed_steps': step,
            'training_time_summary_sha256': queue.sha256(path.parent / 'time_summary.json'),
            'training_seconds': queue.session_totals(path.parent)['training_wall_seconds'],
            'training_wall_time_is_lower_bound': queue.session_totals(path.parent)['wall_time_is_lower_bound']}
            for member, path in enumerate(checkpoints)]
        reports, summary = {}, {}
        for split in splits:
            folder = output / split / 'predictions'
            folder.mkdir(parents=True)
            gid = self.protocol['scenes'][split][0]['global_id']
            prediction = folder / f'{gid}.npz'
            prediction.write_bytes(b'synthetic receipt fixture, never actual evaluation')
            reports[split] = {'map': {'actual_world_counts': [4]}, 'cases': [{'global_id': gid,
                'predictions_sha256': queue.sha256(prediction), 'efficiency': {'actual_world_count': 4}}]}
            summary[split] = {'event_brier': self.score(step), 'map_nll': .4 + step / 1000,
                'map_brier': .2 + step / 100000,
                'collapse_diagnostic': {'generated_nearly_all_free_fraction': 0., 'generated_nearly_all_blocked_fraction': 0.}}
        metrics = {'method': self.method, 'outer_seed': seed, 'stage': stage, 'protocol_sha256': self.digest,
            'splits': reports, 'summary': summary, 'checkpoint_provenance': provenance,
            'observed_and_support_constraints_passed': True, 'final_test_locked': True,
            'location_6_locked': True, 'new_physical_test_images_opened': 0, 'main_table_eligible': False}
        queue.atomic_json(output / 'metrics.json', metrics)
        calibrated = json.loads(calibration_file.read_text()) if calibration_file else {
            'fit_split': 'calibration', 'protocol_sha256': self.digest,
            'prediction_model_identity': queue._model_identity(provenance)}
        queue.atomic_json(output / 'calibration_platt.json', calibrated)
        queue.atomic_json(output / 'complete.json', {'passed': True, 'method': self.method, 'seed': seed,
            'stage': stage, 'completed_steps': [step] * len(checkpoints),
            'metrics_sha256': queue.sha256(output / 'metrics.json')})

    def prepare_stages(self):
        seed = queue.SEEDS[0]
        for stage, step in [('smoke', 1), ('pilot', 5)]:
            checkpoints = [self.train(seed, member, step) for member in range(self.protocol['methods'][self.method]['members'])]
            self.emit_evaluation(self.out / 'stages' / self.method / str(seed) / stage / 'attempt_1',
                                 seed, step, stage, checkpoints, ['calibration', 'validation'])

    def child(self, command, logfile, method, out):
        self.calls.append(command)
        def arg(name):
            return command[command.index(name) + 1]
        seed = int(arg('--seed'))
        if Path(command[1]).name == 'train_flatlands_formal.py':
            step, member = int(arg('--stop-step')), int(arg('--member'))
            self.train(seed, member, step)
            return
        checkpoints = [Path(command[i + 1]) for i, token in enumerate(command) if token == '--checkpoint']
        # All members must exist at this exact stage before evaluating anything.
        members = self.protocol['methods'][method]['members']
        assert len(checkpoints) == members and all(path.exists() for path in checkpoints)
        source = checkpoints[0].read_text()
        step = int(source.split('/')[-1])
        output = Path(arg('--output-dir'))
        if self.fail_once:
            self.fail_once = False
            output.mkdir(parents=True)
            (output / 'partial.txt').write_text('preserve interrupted attempt')
            raise InterruptedError('mock interrupted evaluation')
        stage = arg('--stage')
        split = arg('--split')
        calibration = Path(arg('--calibration-file')) if '--calibration-file' in command else None
        self.emit_evaluation(output, seed, step, stage, checkpoints, [split], calibration)


class FormalFullQueueTests(unittest.TestCase):
    def environment(self, root):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(queue, 'ROOT', root))
        stack.enter_context(mock.patch.object(queue.stage_queue, 'ROOT', root))
        stack.enter_context(mock.patch.object(queue, 'project_path', side_effect=lambda name: root / name))
        queue.stage_queue.STOP.clear()
        stack.enter_context(mock.patch.object(queue, 'SIGNAL_STOP_REQUESTED', False))
        return stack

    def test_three_seed_lama_member_matrix_short_selection_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root, 'lama')
                fixture.prepare_stages()
                with mock.patch.object(queue, 'child', side_effect=fixture.child):
                    pipeline = queue.FullPipeline('lama', fixture.protocol, fixture.protocol_path, fixture.digest,
                                                  fixture.out, fixture.info)
                    result = pipeline.run()
                    self.assertEqual(result['status'], 'all_three_seeds_evaluated_waiting_final_audit')
                    train_calls = [c for c in fixture.calls if Path(c[1]).name == 'train_flatlands_formal.py']
                    self.assertEqual(len(train_calls), (2 + 4 + 4) * 4)
                    eval_calls = [c for c in fixture.calls if Path(c[1]).name == 'evaluate_flatlands_formal.py']
                    final = [c for c in eval_calls if c[c.index('--stage') + 1] == 'formal']
                    self.assertEqual(len(final), 3)
                    self.assertTrue(all(c[c.index('--split') + 1] == 'validation' for c in final))
                    self.assertTrue(all('--calibration-file' in c for c in final))
                    for seed in queue.SEEDS:
                        folder = fixture.out / 'full_runs/lama/seeds' / str(seed)
                        selection = json.loads((folder / 'selection.json').read_text())
                        self.assertEqual(selection['best_step'], 5)
                        self.assertTrue(selection['short_selected_checkpoint'])
                        self.assertFalse(selection['main_table_eligible'])
                        self.assertEqual(len(selection['best_checkpoints']), 4)
                        for path in selection['best_checkpoints']:
                            self.assertTrue((root / path).read_text().endswith('/5'))
                    calls = len(fixture.calls)
                    with self.assertRaises(FileExistsError):
                        queue.FullPipeline('lama', fixture.protocol, fixture.protocol_path, fixture.digest,
                                           fixture.out, fixture.info)
                    resumed = queue.FullPipeline('lama', fixture.protocol, fixture.protocol_path, fixture.digest,
                                                 fixture.out, fixture.info, resume=True)
                    self.assertEqual(resumed.run()['status'], result['status'])
                    self.assertEqual(len(fixture.calls), calls)  # No repeated training OR validation.

    def test_partial_evaluation_kept_and_resume_uses_new_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                fixture.prepare_stages()
                fixture.fail_once = True
                with mock.patch.object(queue, 'child', side_effect=fixture.child):
                    pipeline = queue.FullPipeline('flow', fixture.protocol, fixture.protocol_path, fixture.digest,
                                                  fixture.out, fixture.info)
                    self.assertEqual(pipeline.run()['status'], 'paused')
                    partial = fixture.out / 'evaluations/flow' / str(queue.SEEDS[0]) / 'checkpoint_00000007/attempt_1/partial.txt'
                    self.assertTrue(partial.exists())
                    resumed = queue.FullPipeline('flow', fixture.protocol, fixture.protocol_path, fixture.digest,
                                                 fixture.out, fixture.info, resume=True)
                    self.assertEqual(resumed.run()['status'], 'all_three_seeds_evaluated_waiting_final_audit')
                    self.assertEqual(partial.read_text(), 'preserve interrupted attempt')
                    self.assertTrue(partial.parent.parent.joinpath('attempt_2/complete.json').exists())

    def test_selection_uses_global_tie_set_and_rejects_overwriting_best(self):
        candidates = [{'step': 1000, 'event_brier': .1 + 1.9e-5, 'map_nll': .1},
                      {'step': 5000, 'event_brier': .1 + .9e-5, 'map_nll': .2},
                      {'step': 10000, 'event_brier': .1, 'map_nll': .3}]
        self.assertEqual(queue.select_checkpoint(candidates)['step'], 5000)
        self.assertEqual(queue.select_checkpoint(list(reversed(candidates)))['step'], 5000)
        with self.assertRaisesRegex(ValueError, 'finite'):
            queue.select_checkpoint([{'step': 1000, 'event_brier': float('nan'), 'map_nll': 0.}])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                snapshot = fixture.train(queue.SEEDS[0], 0, 5)
                (snapshot.parent / 'best.pt').write_bytes(b'previous different selection')
                candidate = {'step': 5, 'event_brier': .1, 'map_nll': .2,
                    'checkpoint_provenance': [{'checkpoint': queue.relative(snapshot), 'checkpoint_sha256': queue.sha256(snapshot)}]}
                with self.assertRaisesRegex(FileExistsError, 'never replace'):
                    queue.install_selection(root / 'selection', [candidate], protocol=fixture.protocol,
                        protocol_hash=fixture.digest, method='flow', seed=queue.SEEDS[0], full_plan_hash=fixture.info['sha256'])
                self.assertEqual((snapshot.parent / 'best.pt').read_bytes(), b'previous different selection')

    def test_changed_completed_prediction_or_model_artifact_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                snapshot = fixture.train(queue.SEEDS[1], 0, 7)
                output = fixture.out / 'one_eval'
                fixture.emit_evaluation(output, queue.SEEDS[1], 7, 'checkpoint_calibration', [snapshot], ['calibration'])
                args = dict(protocol=fixture.protocol, protocol_hash=fixture.digest, method='flow', seed=queue.SEEDS[1],
                            step=7, stage='checkpoint_calibration', splits=['calibration'])
                queue.verify_evaluation(output, **args)
                prediction = output / 'calibration/predictions/cal.npz'
                prediction.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'prediction artifact'):
                    queue.verify_evaluation(output, **args)

    def test_held_staged_queue_lock_prevents_full_pipeline_or_child_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                contract = (fixture.protocol_path, fixture.protocol, fixture.digest, fixture.out, {'flow': fixture.info})
                with (fixture.out / '.queue.lock').open('a+b') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with mock.patch.object(queue, 'frozen_contract', return_value=contract), \
                            mock.patch.object(queue, 'FullPipeline') as pipeline, \
                            mock.patch.object(queue.sys, 'argv', ['full.py', '--methods', 'flow']):
                        with self.assertRaises(BlockingIOError):
                            queue.main()
                        pipeline.assert_not_called()

    def test_gate_missing_fails_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                (root / 'data').mkdir()
                (root / 'data/seal.json').write_text('seal')
                fixture.protocol['data_seal_sha256'] = queue.sha256(root / 'data/seal.json')
                fixture.protocol_path.write_text(json.dumps(fixture.protocol))
                fixture.info['path'].unlink()
                with mock.patch.object(queue, 'verify_sources'), mock.patch.object(queue, 'child') as child:
                    with self.assertRaisesRegex(ValueError, 'no approved full plan'):
                        queue.frozen_contract(fixture.protocol_path, ['flow'])
                    child.assert_not_called()
                fixture.protocol['staged'].update(smoke_steps=1000, pilot_steps=5000,
                                                   allowed_full_steps=[10000, 20000])
                fixture.protocol_path.write_text(json.dumps(fixture.protocol))
                plan = {'method': 'flow', 'protocol_sha256': queue.sha256(fixture.protocol_path),
                    'pilot_gate_passed': True, 'full_steps': 10000,
                    'checkpoint_steps': [1000, 5000, 7500, 10000]}
                fixture.info['path'].write_text(json.dumps(plan))
                with mock.patch.object(queue, 'verify_sources'), mock.patch.object(queue, 'child') as child:
                    # Invoke the real frozen driver's gate, not a mock accepting
                    # the Boolean claim. The missing review receipts block it.
                    with self.assertRaisesRegex(ValueError, 'requires'):
                        queue.frozen_contract(fixture.protocol_path, ['flow'])
                    child.assert_not_called()

    def test_resume_refuses_unowned_nonempty_directory_and_changed_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                target = fixture.out / 'full_runs/flow'
                target.mkdir(parents=True)
                old = target / 'old_result.txt'
                old.write_text('preserve old result')
                with self.assertRaisesRegex(ValueError, 'no legitimate'):
                    queue.FullPipeline('flow', fixture.protocol, fixture.protocol_path, fixture.digest,
                                       fixture.out, fixture.info, resume=True)
                self.assertEqual(old.read_text(), 'preserve old result')
                self.assertFalse((target / 'run.json').exists())

    def test_shared_stop_prevents_child_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / 'STOP').touch()
            with mock.patch.object(queue.stage_queue, 'run_child') as call:
                with self.assertRaises(InterruptedError):
                    queue.child(['never-execute'], out / 'log', 'flow', out)
                call.assert_not_called()

    def test_real_sigint_inside_main_thread_lock_and_executor_cleanup(self):
        # Run in a disposable CPU process so a regression cannot hang unittest.
        # This delivers a real POSIX signal while the main thread owns the exact
        # frozen stage queue's non-reentrant lock, then tests cleanup forwarding.
        code = textwrap.dedent('''
            import os, signal, tempfile, threading
            from pathlib import Path
            from unittest import mock
            from scripts import run_flatlands_formal_full as q
            signal.signal(signal.SIGINT, q.request_stop_signal)
            with tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)
                with q.stage_queue.LOCK:
                    os.kill(os.getpid(), signal.SIGINT)
                    assert q.SIGNAL_STOP_REQUESTED
                    assert not q.stage_queue.STOP.is_set()
                q.propagate_stop(out)
                assert q.stage_queue.STOP.is_set()
                print('real_sigint_under_lock_returned', flush=True)
                for cleanup_error in (False, True):
                    q.SIGNAL_STOP_REQUESTED = False
                    q.stage_queue.STOP.clear()
                    released = threading.Event()
                    class Child:
                        pid = 999999
                        sent = False
                        triggered = False
                        def poll(self):
                            if not cleanup_error and not self.triggered:
                                self.triggered = True
                                os.kill(os.getpid(), signal.SIGINT)
                            return 0 if released.is_set() else None
                        def send_signal(self, signum):
                            assert signum == signal.SIGINT
                            self.sent = True
                            released.set()
                    class Pipeline:
                        def run(self):
                            assert released.wait(5), 'stop was not forwarded during join'
                            return {'status': 'paused'}
                    child = Child()
                    q.stage_queue.CHILDREN['flow'] = child
                    if cleanup_error:
                        with mock.patch.object(q, 'atomic_json', side_effect=OSError('injected queue write error')):
                            try:
                                q.run_pipelines([Pipeline()], out, 'digest', ['flow'])
                            except OSError as error:
                                assert str(error) == 'injected queue write error'
                            else:
                                raise AssertionError('original error was swallowed')
                    else:
                        assert q.run_pipelines([Pipeline()], out, 'digest', ['flow']) == [{'status': 'paused'}]
                    assert child.sent
                    q.stage_queue.CHILDREN.clear()
                print('main_loop_and_error_cleanup_forwarded', flush=True)
        ''')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True,
            timeout=20, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT / 'src')})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('real_sigint_under_lock_returned', result.stdout)
        self.assertIn('main_loop_and_error_cleanup_forwarded', result.stdout)

    def test_kill_after_boundary_reconciles_cost_before_validation_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                seed = queue.SEEDS[1]
                checkpoint = fixture.train(seed, 0, 10)
                folder = checkpoint.parent
                previous = (folder / 'time_summary.json').read_bytes()
                # Model boundary 10 already exists; a killed process contributed
                # an extra heartbeat beyond the previously finalized cost.
                queue.atomic_json(folder / 'sessions/killed.started.json', {'session_id': 'killed'})
                queue.atomic_json(folder / 'sessions/killed.heartbeat.json', {'elapsed_seconds': .4})
                pipeline = queue.FullPipeline('flow', fixture.protocol, fixture.protocol_path,
                    fixture.digest, fixture.out, fixture.info)
                with mock.patch.object(queue, 'child', side_effect=fixture.child):
                    evaluation = pipeline.calibration(seed, 10)
                self.assertEqual(len(fixture.calls), 1)
                self.assertEqual(Path(fixture.calls[0][1]).name, 'evaluate_flatlands_formal.py')
                cost = evaluation['metrics']['checkpoint_provenance'][0]
                self.assertAlmostEqual(cost['training_seconds'], 1.4)
                self.assertTrue(cost['training_wall_time_is_lower_bound'])
                audit = queue.training_audit(fixture.out, 'flow', seed, 1, 10, fixture.digest)
                self.assertTrue(audit['passed'])
                self.assertAlmostEqual(audit['training_seconds_sum_across_members'], 1.4)
                self.assertTrue(audit['time_is_lower_bound'])
                archives = list((folder / 'time_reconciliation').glob('previous_summary_*.json'))
                self.assertEqual(len(archives), 1)
                self.assertEqual(archives[0].read_bytes(), previous)
                timing = folder / 'time_summary.json'
                before_mtime = timing.stat().st_mtime_ns
                before_receipt = audit['members'][0]['time_reconciliation']
                again = queue.reconcile_training_time(folder)
                self.assertEqual(timing.stat().st_mtime_ns, before_mtime)
                self.assertEqual(again['receipt'], before_receipt)
                self.assertEqual(len(again['session_files']), 4)
                self.assertEqual(queue.sha256(root / again['receipt']), again['sha256'])

    def test_missing_or_nonfinite_session_cost_fails_without_rewriting_summary(self):
        for bad in (float('nan'), float('inf'), -1, True, '1.0', None):
            with self.subTest(elapsed=bad), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.environment(root):
                    fixture = Fixture(root)
                    folder = fixture.train(queue.SEEDS[0], 0, 10).parent
                    summary = (folder / 'time_summary.json').read_bytes()
                    finish = folder / 'sessions/step_00000010.finished.json'
                    finish.write_text(json.dumps({'elapsed_seconds': bad}))
                    with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                        queue.reconcile_training_time(folder)
                    self.assertEqual((folder / 'time_summary.json').read_bytes(), summary)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                (root / 'time_summary.json').write_text('{"training_wall_seconds": 1.0}')
                with self.assertRaisesRegex(ValueError, 'start receipts are missing'):
                    queue.reconcile_training_time(root)

    def test_failed_then_resumed_session_preserves_failure_without_double_counting_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                folder = fixture.train(queue.SEEDS[0], 0, 10).parent
                sessions = folder / 'sessions'
                queue.atomic_json(sessions / 'failed.started.json', {'session_id': 'failed'})
                queue.atomic_json(sessions / 'failed.heartbeat.json', {'elapsed_seconds': .4})
                queue.atomic_json(sessions / 'failed.finished.json', {'elapsed_seconds': .5, 'status': 'failed'})
                failure = sessions / 'failed.failure.json'
                queue.atomic_json(failure, {'session_id': 'failed', 'wall_seconds_this_session': .45,
                    'category': 'resource', 'error': 'CPU synthetic prior resource failure'})
                original_failure = failure.read_bytes()
                queue.atomic_json(sessions / 'resumed.started.json', {'session_id': 'resumed'})
                queue.atomic_json(sessions / 'resumed.finished.json', {'elapsed_seconds': .7, 'status': 'boundary_complete'})
                reconciled = queue.reconcile_training_time(folder)
                self.assertAlmostEqual(reconciled['totals']['training_wall_seconds'], 2.2)  # 1 + .5 + .7, not + .4/.45
                self.assertFalse(reconciled['totals']['wall_time_is_lower_bound'])
                self.assertIn({'path': queue.relative(failure), 'sha256': queue.sha256(failure)}, reconciled['session_files'])
                self.assertEqual(failure.read_bytes(), original_failure)
                again = queue.reconcile_training_time(folder)
                self.assertEqual(reconciled['receipt'], again['receipt'])
                # A kill during exception handling may leave failure + heartbeat
                # but no finish; its recorded heartbeat is still a lower bound.
                queue.atomic_json(sessions / 'killed.started.json', {'session_id': 'killed'})
                queue.atomic_json(sessions / 'killed.heartbeat.json', {'elapsed_seconds': .2})
                queue.atomic_json(sessions / 'killed.failure.json', {'session_id': 'killed', 'wall_seconds_this_session': .25})
                lower = queue.reconcile_training_time(folder)
                self.assertAlmostEqual(lower['totals']['training_wall_seconds'], 2.4)
                self.assertTrue(lower['totals']['wall_time_is_lower_bound'])
                self.assertEqual(failure.read_bytes(), original_failure)
                queue.atomic_json(sessions / 'orphan.failure.json', {'session_id': 'orphan', 'wall_seconds_this_session': .1})
                with self.assertRaisesRegex(ValueError, 'Orphan or unknown'):
                    queue.reconcile_training_time(folder)

    def test_completed_selected_validation_with_stale_cost_is_preserved_and_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                seed = queue.SEEDS[1]
                checkpoint = fixture.train(seed, 0, 10)
                output = fixture.out / 'final_eval'
                fixture.emit_evaluation(output, seed, 10, 'formal', [checkpoint], ['validation'])
                before = (output / 'metrics.json').read_bytes()
                queue.atomic_json(checkpoint.parent / 'sessions/killed.started.json', {'session_id': 'killed'})
                queue.atomic_json(checkpoint.parent / 'sessions/killed.heartbeat.json', {'elapsed_seconds': .4})
                queue.reconcile_training_time(checkpoint.parent)
                with self.assertRaisesRegex(ValueError, 'stale training cost'):
                    queue.verify_evaluation(output, protocol=fixture.protocol, protocol_hash=fixture.digest,
                        method='flow', seed=seed, step=10, stage='formal', splits=['validation'])
                self.assertEqual((output / 'metrics.json').read_bytes(), before)

    def test_still_improving_is_insufficient_even_with_complete_finite_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                candidates = [{'step': step, 'map_brier': error, 'event_brier': error,
                    'metrics_sha256': str(step), 'collapse_diagnostic': {},
                    'observed_and_support_constraints_passed': True}
                    for step, error in [(5, .3), (7, .2), (10, .1)]]
                selection = {'best_step': 10, 'short_selected_checkpoint': False, 'selected_metrics_sha256': '10'}
                result = queue.convergence_receipt(candidates, {'passed': True, 'errors': []}, selection,
                    protocol=fixture.protocol, protocol_hash=fixture.digest, method='flow', seed=queue.SEEDS[0],
                    full_steps=10, full_plan_hash=fixture.info['sha256'], out=fixture.out)
                self.assertFalse(result['numerical_convergence_passed'])
                self.assertFalse(result['sufficient_training'])
                self.assertIn('event_brier_still_improving_or_unstable', result['insufficiency_reasons'])

    def test_numerical_plateau_needs_bound_visual_review_and_never_grants_main_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                fixture = Fixture(root)
                candidates = [{'step': step, 'map_brier': .1, 'event_brier': .2, 'metrics_sha256': str(step),
                    'collapse_diagnostic': {'generated_nearly_all_free_fraction': 0.},
                    'observed_and_support_constraints_passed': True} for step in (5, 7, 10)]
                selection = {'best_step': 10, 'short_selected_checkpoint': False, 'selected_metrics_sha256': '10'}
                arguments = dict(protocol=fixture.protocol, protocol_hash=fixture.digest, method='flow', seed=queue.SEEDS[0],
                    full_steps=10, full_plan_hash=fixture.info['sha256'], out=fixture.out)
                numerical = queue.convergence_receipt(candidates, {'passed': True, 'errors': []}, selection, **arguments)
                self.assertTrue(numerical['numerical_convergence_passed'])
                self.assertFalse(numerical['sufficient_training'])
                review = fixture.out / 'full_reviews/flow' / f'{queue.SEEDS[0]}.json'
                review.parent.mkdir(parents=True)
                queue.atomic_json(review, {'protocol_sha256': fixture.digest, 'method': 'flow', 'seed': queue.SEEDS[0],
                    'full_plan_sha256': fixture.info['sha256'], 'selected_calibration_metrics_sha256': '10',
                    'last_three_calibration_metrics_sha256': ['5', '7', '10'], 'passed': True, 'no_collapse': True})
                reviewed = queue.convergence_receipt(candidates, {'passed': True, 'errors': []}, selection, **arguments)
                self.assertTrue(reviewed['sufficient_training'])
                self.assertFalse(reviewed['main_table_eligible'])


if __name__ == '__main__':
    unittest.main()
