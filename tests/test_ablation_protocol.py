import unittest
from pathlib import Path

from pathrel.ablation_protocol import ablation_config, checkpoint_action, validate_config


class AblationProtocolTest(unittest.TestCase):
    def setUp(self):
        self.full = {"seed": 7, "decoder_variant": "correlated", "reachability_weight": 2.0,
                     "disable_global_factors": False, "validation_samples": 128,
                     "max_epochs": 40, "patience": 8, "output_dir": "reference", "resume": False}

    def test_only_the_intended_factor_changes(self):
        for variant, key, value in [("no_event", "reachability_weight", 0.0),
                                    ("no_global", "disable_global_factors", True)]:
            config = ablation_config(self.full, variant, "ablation")
            self.assertEqual(config[key], value)
            differences = {k for k in config if config[k] != self.full[k]}
            self.assertEqual(differences, {key, "output_dir"})

    def test_resume_cannot_change_seed_budget_or_second_factor(self):
        expected = ablation_config(self.full, "no_event", "ablation")
        validate_config({**expected, "resume": True, "output_dir": Path("ablation")}, expected)
        for key, value in [("seed", 8), ("validation_samples", 32), ("disable_global_factors", True)]:
            with self.assertRaises(ValueError):
                validate_config({**expected, key: value}, expected)

    def test_stopped_training_resumes_evaluation_without_an_extra_epoch(self):
        config = ablation_config(self.full, "no_global", "ablation")
        state = {"config": config, "epoch": 19, "history": [{"epoch": 19}], "patience_used": 8}
        self.assertEqual(checkpoint_action(state, config), "finalize")
        state["patience_used"] = 7
        self.assertEqual(checkpoint_action(state, config), "resume")
        state.update(epoch=40, history=[{"epoch": 40}])
        self.assertEqual(checkpoint_action(state, config), "finalize")

    def test_nonfull_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            ablation_config({**self.full, "disable_global_factors": True}, "no_event", "ablation")
