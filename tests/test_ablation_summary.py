import math
from types import SimpleNamespace
import unittest

from scripts.build_ablation_summary_zh import interval_conclusion, replay_metrics


def event(scene, probability, target):
    return SimpleNamespace(scene_key=('source', scene), probability=probability, target=target)


class AblationSummaryTest(unittest.TestCase):
    def test_a_large_scene_does_not_outvote_a_small_scene(self):
        rows = [event('a', 1, 0)] + [event('b', 1, 1) for _ in range(9)]
        result = replay_metrics(rows)
        self.assertAlmostEqual(result['brier'], .5)
        self.assertAlmostEqual(result['ece'], .5)
        self.assertAlmostEqual(result['nll'], -.5 * (math.log(1e-6) + math.log(1 - 1e-6)), places=9)
        # Every probability is tied: accepting 30% must retain both scene weights.
        self.assertAlmostEqual(result['risk30'], .5)

    def test_risk_boundary_is_label_independent_and_row_order_invariant(self):
        rows = [event('a', .9, 1), event('a', .8, 1), event('a', .8, 0), event('a', .1, 0)]
        for order in (rows, rows[::-1]):
            # First 25% are correct; the next 5% take equal fractions of a tie.
            self.assertAlmostEqual(replay_metrics(order)['risk30'], .025 / .3)

    def test_ece_uses_global_scene_weights_in_each_bin(self):
        rows = [event('a', .5, 0), event('b', .5, 1)]
        result = replay_metrics(rows)
        self.assertAlmostEqual(result['ece'], 0)
        self.assertAlmostEqual(result['brier'], .25)
        self.assertAlmostEqual(result['nll'], math.log(2))

    def test_null_and_reversed_intervals_cannot_be_called_a_stable_full_model_win(self):
        for intervals in ([(0, .1)] * 3, [(-.1, .1)] * 3, [(.1, .2), (.1, .2), (-.2, -.1)]):
            self.assertIn('尚未证实', interval_conclusion(intervals))
        self.assertIn('反而更好', interval_conclusion([(-.2, -.1)] * 3))
        self.assertIn('支持完整模型', interval_conclusion([(.1, .2)] * 3))

    def test_invalid_or_empty_predictions_are_rejected(self):
        for records in ([], [event('a', float('nan'), 1)], [event('a', 1.1, 1)], [event('a', .5, .5)]):
            with self.assertRaises(ValueError):
                replay_metrics(records)
