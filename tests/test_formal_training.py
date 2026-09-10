import unittest
import torch

from pathrel.formal_training import StagedLRScheduler, weighted_micro_loss


class FormalTrainingContractTests(unittest.TestCase):
    def test_zero_query_scene_does_not_reweight_microbatch_events(self):
        # Four parents, one without a query. Unbalanced micros contain 1 and 2
        # query parents. This used to weight event means 1/2 each incorrectly.
        parameter = torch.tensor(2., requires_grad=True)
        first = weighted_micro_loss(parameter, parameter*0, parameter*3,
            micro_count=2, batch_count=4, micro_query_scenes=1, batch_query_scenes=3,
            event_weight=2., variogram_weight=.1)
        second = weighted_micro_loss(parameter*4, parameter*0, parameter*9,
            micro_count=2, batch_count=4, micro_query_scenes=2, batch_query_scenes=3,
            event_weight=2., variogram_weight=.1)
        expected = (parameter+parameter*4)/2 + 2*(parameter*3+2*parameter*9)/3
        torch.testing.assert_close(first+second, expected)
        actual_grad = torch.autograd.grad(first+second, parameter, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, parameter)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_schedule_frozen_after_pilot_and_resume_is_exact(self):
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        initial = StagedLRScheduler(opt, pilot_steps=5)
        for _ in range(5):
            initial.step()
        self.assertEqual(opt.param_groups[0]['lr'], 1e-4)
        state = initial.state_dict()
        full = StagedLRScheduler(opt, pilot_steps=5, full_steps=10)
        full.load_state_dict(state)
        for _ in range(2):
            full.step()
        state = full.state_dict()
        fresh_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        resumed = StagedLRScheduler(fresh_opt, pilot_steps=5, full_steps=10)
        resumed.load_state_dict(state)
        for _ in range(3):
            full.step(); resumed.step()
            self.assertEqual(opt.param_groups[0]['lr'], fresh_opt.param_groups[0]['lr'])
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 1e-5)
        changed = StagedLRScheduler(fresh_opt, pilot_steps=5, full_steps=20)
        with self.assertRaises(ValueError):
            changed.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
