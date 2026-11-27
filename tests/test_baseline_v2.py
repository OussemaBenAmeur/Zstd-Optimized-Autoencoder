import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from jpeg_reference import compare, interpolate, summarize
from zae.data import TrainingCrops, decode_all
from zae.tracking import validate_config


def configs(name):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        return compose(config_name=name)


class BaselineV2Tests(unittest.TestCase):
    def test_cached_crops_are_identical_to_decoded_crops(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for index, shape in enumerate([(40, 52, 3), (48, 36, 3), (33, 64, 3)]):
                files.append(Path(directory) / f"{index}.png")
                Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(files[-1])
            plain = TrainingCrops(files, 32, 12, seed=5, flip=True)
            cached = TrainingCrops(files, 32, 12, seed=5, flip=True, images=decode_all(files, 2))
            for index in range(len(plain)):
                torch.testing.assert_close(plain[index], cached[index], rtol=0, atol=0)
                self.assertEqual(cached[index].dtype, torch.float32)

    def test_v2_config_is_valid_and_v1_is_unchanged(self):
        v2 = configs("baseline_v2")
        validate_config(v2)
        self.assertTrue(v2.data.cache_images)
        self.assertEqual(v2.runtime.precision, "bfloat16_mixed")
        self.assertIsNone(v2.quantization.calibration_crop_size)
        v1 = configs("config")
        self.assertFalse(v1.data.cache_images)
        self.assertEqual(v1.runtime.precision, "float32")
        self.assertEqual(v1.quantization.calibration_crop_size, 256)
        self.assertEqual(v1.training.learning_rate, 0.0002)
        # v2 changes only training settings that the resume guard protects, not the model.
        self.assertEqual(dict(v1.model), dict(v2.model))

    def test_v3_config_inherits_v2_and_validates(self):
        v3 = configs("baseline_v3")
        validate_config(v3)
        self.assertTrue(v3.runtime.compile)
        self.assertTrue(v3.data.cache_images)
        self.assertEqual(v3.training.learning_rate, 0.0003)
        self.assertEqual(v3.training.skip_update_grad_norm, 100.0)
        self.assertIsNone(configs("config").training.skip_update_grad_norm)
        cfg = configs("baseline_v3")
        cfg.training.skip_update_grad_norm = 0
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_invalid_runtime_options_are_rejected(self):
        for key, value in [("runtime.precision", "float16"), ("data.cache_images", "yes"),
                           ("quantization.calibration_crop_size", 0)]:
            with self.subTest(key=key):
                cfg = configs("baseline_v2")
                section, field = key.split(".")
                cfg[section][field] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_jpeg_summary_and_interpolation(self):
        rows = []
        for quality, size, quality_db in [(10, 100, 25.0), (50, 300, 30.0), (90, 900, 40.0)]:
            for side in (10, 20):
                rows.append({"quality": quality, "height": side, "width": side, "jpeg_bytes": size,
                             "psnr_db": quality_db, "ssim": 0.9})
        curve = summarize(rows)
        self.assertAlmostEqual(curve[0]["bpp"], 8 * 200 / 500)
        self.assertAlmostEqual(interpolate(curve, "bpp", "psnr_mean", 6.4), 27.5)
        self.assertIsNone(interpolate(curve, "bpp", "psnr_mean", 100))
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_text(json.dumps({"int8": {"bpp_after_zstd": 3.2, "psnr_mean": 25.0, "ssim_mean": 0.8}}))
            result = compare(curve, {"ae": summary})["ae"]
        self.assertAlmostEqual(result["jpeg_psnr_at_same_bpp"], 25.0)
        self.assertAlmostEqual(result["psnr_gap_db"], 0.0)
        self.assertAlmostEqual(result["bits_ratio_vs_jpeg"], 1.0)


if __name__ == "__main__":
    unittest.main()
