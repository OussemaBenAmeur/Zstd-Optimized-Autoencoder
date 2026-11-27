import unittest

from hydra import compose, initialize_config_dir
from pathlib import Path

from zae.early_stopping import EarlyStopping
from zae.tracking import validate_config


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_exactly_patience_checks(self):
        stopping = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
        self.assertFalse(stopping.update(20.0))  # Establish reference, not a bad check.
        self.assertFalse(stopping.update(20.0))
        self.assertTrue(stopping.update(19.9))
        self.assertEqual(stopping.bad_checks, 2)

    def test_significant_improvement_resets_patience(self):
        stopping = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
        for score in [20.0, 20.005, 20.02]:
            self.assertFalse(stopping.update(score))
        self.assertEqual(stopping.reference_psnr, 20.02)
        self.assertEqual(stopping.bad_checks, 0)
        self.assertFalse(stopping.update(20.02))

    def test_small_gains_accumulate_against_last_significant_reference(self):
        stopping = EarlyStopping(enabled=True, patience=3, min_delta=1.0)
        for score in [20.0, 20.4, 20.8, 21.1]:
            self.assertFalse(stopping.update(score))
        self.assertEqual(stopping.reference_psnr, 21.1)
        self.assertEqual(stopping.bad_checks, 0)

    def test_threshold_is_strict_and_zero_delta_still_requires_a_gain(self):
        for delta in [0.0, 1.0]:
            with self.subTest(delta=delta):
                stopping = EarlyStopping(enabled=True, patience=1, min_delta=delta)
                stopping.update(20.0)
                self.assertTrue(stopping.update(20.0 + delta))

    def test_disabled_never_stops(self):
        stopping = EarlyStopping(enabled=False, patience=1, min_delta=0.01)
        for score in [20.0, 19.0, 18.0, 17.0]:
            self.assertFalse(stopping.update(score))

    def test_checkpoint_preserves_patience(self):
        stopping = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
        stopping.update(20.0)
        stopping.update(20.005)
        resumed = EarlyStopping(enabled=True, patience=2, min_delta=0.01)
        resumed.load_state_dict(stopping.state_dict())
        self.assertTrue(resumed.update(20.009))
        self.assertEqual(resumed.state_dict(), {"reference_psnr": 20.0, "bad_checks": 2, "stopped": True})

    def test_invalid_config_and_nonfinite_metrics_are_rejected(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
            cfg = compose(config_name="config")
        for field, invalid in [("patience", 0), ("patience", 1.5), ("min_delta", -0.01),
                               ("min_delta", float("nan")), ("min_delta", float("inf")),
                               ("enabled", "yes")]:
            with self.subTest(field=field, invalid=invalid):
                saved = cfg.training.early_stopping[field]
                cfg.training.early_stopping[field] = invalid
                with self.assertRaises(ValueError):
                    validate_config(cfg)
                cfg.training.early_stopping[field] = saved
        for invalid in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(ValueError):
                EarlyStopping(True, 2, 0.01).update(invalid)


if __name__ == "__main__":
    unittest.main()
