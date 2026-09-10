import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_flatlands_formal_stages as queue


class FormalQueueTests(unittest.TestCase):
    def test_partial_and_wrong_method_evaluations_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder)
            attempt=parent/'attempt_1'; attempt.mkdir()
            metrics={'protocol_sha256':'a'*64,'stage':'pilot','method':'flow','outer_seed':20260831}
            (attempt/'metrics.json').write_text(json.dumps(metrics))
            self.assertIsNone(queue.find_finished_eval(parent,'a'*64,'pilot','lama',20260831,5000))
            (attempt/'complete.json').write_text(json.dumps({'passed':True,'metrics_sha256':queue.sha256(attempt/'metrics.json')}))
            with self.assertRaisesRegex(ValueError,'identity'):
                queue.find_finished_eval(parent,'a'*64,'pilot','lama',20260831,5000)

    def test_worker_failure_is_recorded_before_any_next_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            protocol={'output_root':'out','data_root':'data','staged':{'seed':20260831,'smoke_steps':1000,'pilot_steps':5000},
                      'methods':{'flow':{'members':1}}}
            path=root/'protocol.json';path.write_text(json.dumps(protocol))
            with patch.object(queue,'ROOT',root), patch.object(queue,'run_child',side_effect=RuntimeError('test worker failure')) as run:
                result=queue.method_stages('flow',protocol,path)
            self.assertEqual(run.call_count,1)
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['full_training_started'])
            self.assertEqual(json.loads((root/'out/stage_status/flow.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()
