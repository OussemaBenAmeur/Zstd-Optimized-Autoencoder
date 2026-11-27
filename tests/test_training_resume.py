"""Real CLI regression: interrupted training must match uninterrupted CPU training."""
import os
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
            environment = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}

            def run(command, output, extra=()):
                return subprocess.run([sys.executable, *command, *common,
                                       f"hydra.run.dir={root / output}", *extra],
                                      cwd=project, env=environment, capture_output=True, text=True,
                                      timeout=90)

            full = run(["train.py"], "full")
            self.assertEqual(full.returncode, 0, full.stdout + full.stderr)
            # Interrupt after a real optimizer/validation/checkpoint cycle. Production code
            # has no test-only stopping mode, and resume uses the unchanged max_steps.
            interruption = """
import train
original = train.save_checkpoint
def save_and_interrupt(path, model, optimizer, step, *args):
    original(path, model, optimizer, step, *args)
    if path.name == 'last.pt' and step == 1:
        raise RuntimeError('intentional interruption for resume test')
train.save_checkpoint = save_and_interrupt
train.main()
"""
            interrupted = run(["-c", interruption], "interrupted")
            self.assertNotEqual(interrupted.returncode, 0)
            self.assertIn("intentional interruption", interrupted.stderr)
            resumed = run(["train.py"], "resumed", [
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


if __name__ == "__main__":
    unittest.main()
