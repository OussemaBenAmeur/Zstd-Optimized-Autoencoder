import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from evaluate import summarize
from train import check_resume, learning_rate
from zae.codec import deserialize, latent_tensor, quantize, serialize
from zae.metrics import encode_padded, psnr, round_rgb, ssim
from zae.model import AutoEncoder
from zae.tracking import validate_config


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
            self.cfg = compose(config_name="smoke")

    def test_float_and_quantized_files_restore_identical_latent_bytes(self):
        z = torch.randn(1, 4, 3, 5)
        scales = torch.tensor([0.01, 0.02, 0.03, 0.04]).reshape(1, 4, 1, 1)
        for quantized in [False, True]:
            tensor = quantize(z, scales) if quantized else z
            array = tensor[0].numpy()
            metadata = {"quantized": quantized, "scales": scales.flatten().tolist() if quantized else None}
            plain = serialize(array, metadata)
            compressed = serialize(array, metadata, self.cfg.zstd)
            restored, header = deserialize(compressed)
            original, original_header = deserialize(plain)
            self.assertEqual(array.tobytes(), restored.tobytes())
            self.assertEqual(original_header, header)
            expected = tensor.float() * scales if quantized else z
            torch.testing.assert_close(latent_tensor(restored, header, "cpu"), expected, rtol=0, atol=0)
            self.assertGreater(len(plain), array.nbytes)  # Side information counts toward size.

    def test_odd_image_dimensions_and_serialized_reconstruction(self):
        model = AutoEncoder(hidden_channels=8, latent_channels=4).eval()
        x = torch.rand(1, 3, 35, 47)
        with torch.inference_mode():
            z = encode_padded(model, x)
            self.assertEqual(tuple(z.shape), (1, 4, 3, 3))
            blob = serialize(z[0].numpy(), {"quantized": False, "scales": None}, self.cfg.zstd)
            restored, metadata = deserialize(blob)
            output = model.decode(latent_tensor(restored, metadata, "cpu"))[..., :35, :47]
            reference = model.decode(z)[..., :35, :47]
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            self.assertEqual(output.shape, x.shape)

    def test_metric_definitions(self):
        x = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(ssim(x, x), torch.ones(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(psnr(x, x + 0.01), torch.full((2,), 40.0), atol=0.0001, rtol=0)
        self.assertTrue(torch.isinf(psnr(x, x)).all())
        self.assertTrue((round_rgb(torch.tensor([-1., 2.])) == torch.tensor([0., 1.])).all())

    def test_full_and_smoke_configs_and_schedule(self):
        validate_config(self.cfg)
        with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
            baseline = compose(config_name="config")
        validate_config(baseline)
        self.assertEqual(baseline.model.hidden_channels, 128)
        self.assertEqual(baseline.model.latent_channels, 64)
        self.assertEqual(baseline.training.max_steps, 40000)
        t = baseline.training
        self.assertAlmostEqual(learning_rate(t.warmup_steps - 1, t), t.learning_rate)
        self.assertAlmostEqual(learning_rate(t.max_steps - 1, t), t.minimum_learning_rate)

    def test_resume_rejects_experiment_or_data_changes(self):
        checkpoint = {"config": OmegaConf.to_container(self.cfg, resolve=True), "manifest": {"id": "same"}}
        check_resume(checkpoint, self.cfg, {"id": "same"})
        changed = OmegaConf.create(OmegaConf.to_container(self.cfg))
        changed.model.latent_channels = 8
        with self.assertRaises(ValueError):
            check_resume(checkpoint, changed, {"id": "same"})
        with self.assertRaises(ValueError):
            check_resume(checkpoint, self.cfg, {"id": "changed"})

    def test_summary_uses_total_bytes_and_total_pixels(self):
        rows = []
        for side, size in [(10, 100), (20, 200)]:
            row = {"branch": "float32", "height": side, "width": side, "psnr_db": 30., "ssim": 0.9}
            for column in ["original_png_bytes", "raw_rgb_bytes", "latent_payload_bytes", "serialized_bytes",
                           "zstd_file_bytes", "reconstruction_png_bytes"]:
                row[column] = size
            rows.append(row)
        self.assertAlmostEqual(summarize(rows)["float32"]["bpp_after_zstd"], 8 * 300 / 500)


if __name__ == "__main__":
    unittest.main()
