"""Real CLI regression: interrupted training must match uninterrupted CPU training."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


class ResumeIntegrationTest(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted(self):
        self.check_continuation(early_stopping=False)

    def test_early_stopping_and_interrupted_patience_match_uninterrupted(self):
        self.check_continuation(early_stopping=True)

    def test_zstd_loss_resume_and_frozen_quantizer_evaluation(self):
        self.check_continuation(early_stopping=False, zstd=True)

    def test_zstd_spike_guard_preserves_weights_optimizer_and_resume(self):
        self.check_continuation(early_stopping=False, zstd=True, skip_updates=True)

    def check_continuation(self, early_stopping, zstd=False, skip_updates=False):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="zae-resume-") as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "test").mkdir()
            rng = np.random.default_rng(7)
            for folder, names in [("train", ["a", "b", "c"]), ("test", ["d"])]:
                for name in names:
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                        root / folder / (name + ".png"))
            common = [
                "--config-name", "smoke", f"data.root={root}", "data.train_dirs=[train]",
                "data.test_dir=test", "data.expected_train_images=3", "data.expected_test_images=1",
                "data.validation_images=1", f"tracking.uri=sqlite:///{root / 'mlflow.db'}",
                f"tracking.artifact_location={root / 'artifacts'}",
            ]
            if early_stopping:
                common += ["training.max_steps=8", "training.early_stopping.enabled=true",
                           "training.early_stopping.patience=2", "tracking.log_every_steps=4"]
            if zstd:
                from hydra import compose, initialize_config_dir
                from zae.data import manifest
                with initialize_config_dir(version_base="1.3", config_dir=str(project / "configs")):
                    cfg = compose(config_name="smoke", overrides=common[2:8])
                data = manifest(cfg)
                calibration = {"scheme": cfg.quantization.scheme, "dtype": "int8", "qmax": 127,
                               "scales": [0.01] * 4, "dataset_id": data["dataset_id"],
                               "percentile": 99.9, "crop_size": 32, "images": data["train"]}
                path = root / "quantization.json"
                path.write_text(json.dumps(calibration))
                # Exercise the actual v3-derived config, preserving its cached crops,
                # channels_last, compilation and spike guard. Only size/device/budget
                # are reduced for this synthetic CPU regression.
                common[1] = "zstd"
                common += [f"quantization.calibration_file={path}", "experiment.name=smoke",
                           "model.hidden_channels=8", "model.latent_channels=4", "data.crop_size=32",
                           "training.batch_size=1", "training.max_steps=2", "training.warmup_steps=0",
                           "training.early_stopping.enabled=false", "validation.every_steps=1",
                           "validation.crop_size=32", "runtime.device=cpu", "runtime.precision=float32",
                           "runtime.num_workers=0", "runtime.cpu_threads=2", "tracking.system_metrics=false",
                           "tracking.log_every_steps=2", "tracking.reconstruction_samples=2",
                           "evaluation.split=validation", "evaluation.max_images=2", "evaluation.full_resolution=false"]
                if skip_updates:
                    common += ["training.skip_update_grad_norm=1e-12"]
            environment = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}

            def run(command, output, extra=()):
                return subprocess.run([sys.executable, *command, *common,
                                       f"hydra.run.dir={root / output}", *extra],
                                      cwd=project, env=environment, capture_output=True, text=True,
                                      timeout=90)

            # A flat synthetic validation score makes the stopping point predictable.
            validation_patch = ("train.validate = lambda *args: {'val_psnr': 20.0, 'val_ssim': 0.9}\n"
                                if early_stopping else "")
            # Trace the full compiled objective without requiring a GPU compiler toolchain.
            compile_patch = ("original_compile = train.torch.compile\n"
                             "train.torch.compile = lambda module: original_compile(module, backend='eager', fullgraph=True)\n"
                             if zstd else "")
            entry = ["-c", "import train\n" + compile_patch + validation_patch + "train.main()\n"]
            full = run(entry, "full")
            self.assertEqual(full.returncode, 0, full.stdout + full.stderr)
            # Interrupt after a real optimizer/validation/checkpoint cycle. Production code
            # has no test-only stopping mode, and resume uses the unchanged max_steps.
            interruption = """
import train
original = train.save_checkpoint
def save_and_interrupt(path, model, optimizer, step, *args):
    original(path, model, optimizer, step, *args)
    if path.name == 'last.pt' and step == INTERRUPT_STEP:
        raise RuntimeError('intentional interruption for resume test')
train.save_checkpoint = save_and_interrupt
"""
            interruption = interruption.replace("INTERRUPT_STEP", "2" if early_stopping else "1")
            interruption += compile_patch + validation_patch + "train.main()\n"
            interrupted = run(["-c", interruption], "interrupted")
            self.assertNotEqual(interrupted.returncode, 0)
            self.assertIn("intentional interruption", interrupted.stderr)
            resumed = run(entry, "resumed", [
                f"training.resume_checkpoint={root / 'interrupted' / 'last.pt'}"])
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            full_checkpoint = torch.load(root / "full" / "last.pt", map_location="cpu", weights_only=False)
            resumed_checkpoint = torch.load(root / "resumed" / "last.pt", map_location="cpu", weights_only=False)
            interrupted_checkpoint = torch.load(root / "interrupted" / "last.pt", map_location="cpu", weights_only=False)
            for key, weights in full_checkpoint["model"].items():
                torch.testing.assert_close(weights, resumed_checkpoint["model"][key], rtol=0, atol=0)
            self.assertEqual(full_checkpoint["history"], resumed_checkpoint["history"])
            self.assertEqual(interrupted_checkpoint["run_id"], resumed_checkpoint["run_id"])
            self.assertTrue((root / "resumed" / "best.pt").exists())
            self.assertEqual(full_checkpoint["early_stopping"], resumed_checkpoint["early_stopping"])
            self.assertEqual(full_checkpoint["step"], 3 if early_stopping else 2)
            if zstd:
                self.assertEqual(resumed_checkpoint["quantization_calibration"], calibration)
                self.assertIn("loss_components", interrupted_checkpoint["training_log_state"])
                self.assertEqual(resumed_checkpoint["training_log_state"]["skipped_updates"], 2 if skip_updates else 0)
                if skip_updates:
                    self.assertFalse(resumed_checkpoint["optimizer"]["state"])
                    from zae.model import AutoEncoder
                    torch.manual_seed(0)
                    initial = AutoEncoder(hidden_channels=8, latent_channels=4).state_dict()
                    for name, value in initial.items():
                        torch.testing.assert_close(value, resumed_checkpoint["model"][name], rtol=0, atol=0)
                evaluated = run(["evaluate.py"], "evaluation", [
                    f"evaluation.checkpoint={root / 'resumed' / 'last.pt'}"])
                self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
                evaluation = json.loads((root / "evaluation" / "summary.json").read_text())["int8"]
                final_validation = [r for r in resumed_checkpoint["history"] if "val_zstd_bpp" in r][-1]
                self.assertEqual(evaluation["bpp_after_zstd"], final_validation["val_zstd_bpp"])
                self.assertEqual(evaluation["psnr_mean"], final_validation["val_psnr"])
                train_record = next(r for r in resumed_checkpoint["history"] if "train_rate_proxy_bpp" in r)
                self.assertGreater(train_record["train_loss"], train_record["train_mse"])
            for output in ["full", "resumed"]:
                summary = json.loads((root / output / "training_summary.json").read_text())
                self.assertEqual(summary["steps"], 3 if early_stopping else 2)
                self.assertEqual(summary["max_steps"], 8 if early_stopping else 2)
                self.assertEqual(summary["stopped_early"], early_stopping)
                self.assertEqual(summary["stop_reason"], "early_stopping" if early_stopping else "max_steps")
                training_rows = [r for r in resumed_checkpoint["history"] if "train_mse" in r]
                self.assertEqual(training_rows[-1]["step"], summary["steps"])
            if early_stopping:
                self.assertEqual(interrupted_checkpoint["early_stopping"]["bad_checks"], 1)
                self.assertEqual(resumed_checkpoint["early_stopping"]["bad_checks"], 2)
                self.assertEqual(len(training_rows), 1)  # Flush at step 3, before log interval 4.
                from mlflow import MlflowClient
                client = MlflowClient(tracking_uri=f"sqlite:///{root / 'mlflow.db'}")
                run_record = client.get_run(resumed_checkpoint["run_id"])
                self.assertEqual(run_record.data.tags["stop_reason"], "early_stopping")
                self.assertEqual(run_record.data.metrics["completed_steps"], 3)
                terminal_resume = run(entry, "terminal", [
                    f"training.resume_checkpoint={root / 'resumed' / 'last.pt'}"])
                self.assertNotEqual(terminal_resume.returncode, 0)
                self.assertIn("already early-stopped", terminal_resume.stderr)
                # If patience expires on the final permitted update, the actual reason
                # is the step cap: training did not finish before its configured budget.
                capped = run(entry, "capped", ["training.max_steps=3"])
                self.assertEqual(capped.returncode, 0, capped.stdout + capped.stderr)
                capped_summary = json.loads((root / "capped" / "training_summary.json").read_text())
                self.assertEqual(capped_summary["steps"], 3)
                self.assertEqual(capped_summary["stop_reason"], "max_steps")
                self.assertFalse(capped_summary["stopped_early"])


if __name__ == "__main__":
    unittest.main()
