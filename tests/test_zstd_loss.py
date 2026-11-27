import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from evaluate import calibrate
from train import check_resume, validate
from zae.codec import quantize, serialize
from zae.losses import ZstdRateLoss, zstd_file_sizes
from zae.model import AutoEncoder
from zae.tracking import validate_config


class ZstdLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
            self.cfg = compose(config_name="smoke")
            self.zstd_cfg = compose(config_name="zstd")
        self.scales = torch.full((1, 4, 1, 1), 0.01)
        self.objective = ZstdRateLoss(self.scales, 127, 0.001, 0.1, 0.01)
        self.calibration = {"scheme": self.cfg.quantization.scheme, "dtype": "int8", "qmax": 127,
                            "scales": [0.01] * 4, "dataset_id": "test-dataset"}

    def test_same_architecture_and_valid_config(self):
        validate_config(self.zstd_cfg)
        with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
            baseline = compose(config_name="config")
        self.assertEqual(baseline.model, self.zstd_cfg.model)
        self.assertEqual(baseline.data, self.zstd_cfg.data)
        for field in ("batch_size", "max_steps", "learning_rate", "warmup_steps", "early_stopping"):
            self.assertEqual(baseline.training[field], self.zstd_cfg.training[field])
        for invalid in (-1., float("nan"), float("inf"), True):
            cfg = copy.deepcopy(self.zstd_cfg)
            cfg.training.zstd_loss.lambda_rate = invalid
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_exact_int8_forward_and_surrogate_gradient(self):
        z = (torch.randn(1, 4, 8, 8) * 2).requires_grad_()
        output = self.objective.quantized_latent(z)
        expected = quantize(z, self.scales).float() * self.scales
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(z.grad, torch.ones_like(z), rtol=0, atol=0)

    def test_rate_penalty_prefers_repetition_with_same_histogram(self):
        ordered = torch.arange(-16, 16).float().repeat_interleave(128)
        shuffled = ordered[torch.randperm(ordered.numel())]
        scales = torch.ones(1, 1, 1, 1)
        loss = ZstdRateLoss(scales, 127, 0.001, 0.1, 0.01)
        a, b = ordered.reshape(1, 1, 64, 64), shuffled.reshape(1, 1, 64, 64)
        ra, rb = loss.rate_terms(a, 4096), loss.rate_terms(b, 4096)
        torch.testing.assert_close(ra["entropy_bpp"], rb["entropy_bpp"])
        self.assertLess(ra["spatial_proxy_bpp"], rb["spatial_proxy_bpp"])
        self.assertLess(zstd_file_sizes(a, scales, 127, 64, 64, self.cfg.zstd)[0],
                        zstd_file_sizes(b, scales, 127, 64, 64, self.cfg.zstd)[0])
        constant = loss.rate_terms(torch.zeros_like(a), 4096)
        self.assertEqual(constant["rate_proxy_bpp"].item(), 0)

    def test_loss_reaches_encoder_and_decoder_and_handles_clipping(self):
        model = AutoEncoder(**OmegaConf.to_container(self.cfg.model))
        loss, components = self.objective(model, torch.rand(2, 3, 32, 32))
        self.assertAlmostEqual(loss.item(), (components["mse"] + 0.001 * components["rate_proxy_bpp"]).item())
        loss.backward()
        for module in (model.encoder, model.decoder):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        z = torch.full((1, 4, 2, 2), 2., requires_grad=True)
        self.objective.rate_terms(z, 1024)["rate_proxy_bpp"].backward()
        self.assertTrue((z.grad > 0).all())  # Overflow values are pushed back toward the int8 range.

    def test_zero_lambda_is_quantization_aware_mse_control(self):
        model = AutoEncoder(**OmegaConf.to_container(self.cfg.model))
        objective = ZstdRateLoss(self.scales, 127, 0., 0.1, 0.01)
        loss, terms = objective(model, torch.rand(1, 3, 32, 32))
        torch.testing.assert_close(loss, terms["mse"], rtol=0, atol=0)

    def test_size_monitor_matches_complete_evaluation_files(self):
        z = torch.randn(2, 4, 3, 5)
        sizes = zstd_file_sizes(z, self.scales, 127, 35, 65, self.cfg.zstd)
        for array, size in zip(quantize(z, self.scales).numpy(), sizes):
            metadata = {"image_height": 35, "image_width": 65, "quantized": True,
                        "scales": self.scales.flatten().tolist(), "checkpoint_sha256": "a" * 64}
            self.assertEqual(size, len(serialize(array, metadata, self.cfg.zstd)))

    def test_validation_uses_quantized_decoder_and_real_zstd(self):
        model = AutoEncoder(**OmegaConf.to_container(self.cfg.model))
        images = [torch.rand(3, 32, 32), torch.rand(3, 32, 32)]
        record = validate(model, images, 1, "cpu", self.objective, self.cfg.zstd)
        with torch.inference_mode():
            z = model.encode(torch.stack(images))
            sizes = zstd_file_sizes(z, self.scales, 127, 32, 32, self.cfg.zstd)
        self.assertEqual(record["val_zstd_file_bytes"], sum(sizes))
        self.assertEqual(record["val_zstd_bpp"], 8 * sum(sizes) / (2 * 32 * 32))
        self.assertAlmostEqual(record["val_rate_distortion"],
                               record["val_mse"] + 0.001 * record["val_zstd_bpp"])

    def test_evaluation_preserves_checkpoint_quantizer_without_recalibration(self):
        checkpoint = {"config": {"training": {"loss": "mse_zstd"}},
                      "quantization_calibration": self.calibration}
        scales, calibration = calibrate(self.cfg, None, {"dataset_id": "test-dataset"},
                                         "cpu", "hash", checkpoint)
        torch.testing.assert_close(scales, self.scales)
        self.assertEqual(calibration, self.calibration)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            path.write_text(json.dumps({**self.calibration, "scales": [0.02] * 4}))
            self.cfg.quantization.calibration_file = str(path)
            with self.assertRaisesRegex(ValueError, "differs from the frozen"):
                calibrate(self.cfg, None, {"dataset_id": "test-dataset"}, "cpu", "hash", checkpoint)

    def test_resume_rejects_changed_calibration_and_zstd_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            path.write_text(json.dumps(self.calibration))
            cfg = self.zstd_cfg
            cfg.model.latent_channels = 4
            cfg.quantization.calibration_file = str(path)
            data = {"dataset_id": "test-dataset"}
            checkpoint = {"config": OmegaConf.to_container(cfg, resolve=True), "manifest": data,
                          "quantization_calibration": self.calibration}
            check_resume(checkpoint, cfg, data)
            cfg.zstd.level = 5
            with self.assertRaisesRegex(ValueError, "changed zstd"):
                check_resume(checkpoint, cfg, data)
            cfg.zstd.level = 3
            path.write_text(json.dumps({**self.calibration, "scales": [0.02] * 4}))
            with self.assertRaisesRegex(ValueError, "Frozen calibration changed"):
                check_resume(checkpoint, cfg, data)


if __name__ == "__main__":
    unittest.main()
