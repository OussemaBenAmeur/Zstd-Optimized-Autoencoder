import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from zae.codec import quantize, serialize
from zae.losses import ZstdSizeLoss
from zae.model import AutoEncoder
from zae.tracking import validate_config


class ZstdLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
            self.cfg = compose(config_name="zstd")
            self.baseline = compose(config_name="baseline_v3")
        self.scales = torch.full((1, 4, 1, 1), 0.01)
        self.objective = ZstdSizeLoss(self.scales, 127, self.cfg.zstd)

    def test_only_loss_settings_differ_from_v3(self):
        validate_config(self.cfg)
        expected = OmegaConf.to_container(self.baseline, resolve=True)
        actual = OmegaConf.to_container(self.cfg, resolve=True)
        actual["experiment"] = expected["experiment"]
        actual["training"].pop("zstd_loss")
        actual["training"]["loss"] = "mse"
        self.assertEqual(actual, expected)
        self.assertIsNone(self.cfg.quantization.calibration_file)
        for field, values in {"lambda_rate": [-1, float("nan"), float("inf"), True],
                              "directions": [0, 1.5, True],
                              "perturbation_symbols": [0, -1, float("nan")]}.items():
            for value in values:
                cfg = copy.deepcopy(self.cfg)
                cfg.training.zstd_loss[field] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_forward_equals_real_evaluation_codec_file_sizes(self):
        z = torch.randn(2, 4, 8, 8) * 0.3
        metadata = {"image_height": 125, "image_width": 127, "quantized": True,
                    "scales": self.scales.flatten().tolist(), "checkpoint_sha256": "a" * 64}
        files = [serialize(a, metadata, self.cfg.zstd) for a in quantize(z, self.scales).numpy()]
        expected = torch.tensor([8 * len(blob) / (125 * 127) for blob in files]).mean()
        torch.testing.assert_close(self.objective(z, 125, 127), expected, rtol=0, atol=0)

    def test_larger_zstd_files_raise_the_loss_for_identical_mse(self):
        ordered = torch.arange(-16, 16).float().repeat_interleave(128)
        shuffled = ordered[torch.randperm(ordered.numel())]
        objective = ZstdSizeLoss([1.0], 127, self.cfg.zstd)
        a, b = ordered.reshape(1, 1, 64, 64), shuffled.reshape(1, 1, 64, 64)
        mse, weight = 0.01, 0.001
        self.assertLess(mse + weight * objective(a, 1024, 1024),
                        mse + weight * objective(b, 1024, 1024))

    def test_size_gradient_points_toward_the_probe_with_smaller_real_files(self):
        compact = torch.zeros(1, 1, 32, 32)
        large = torch.randint(-32, 33, compact.shape).float()
        z = ((compact + large) / 2).requires_grad_()
        direction = (compact - large) / 2
        objective = ZstdSizeLoss([1.0], 127, self.cfg.zstd, directions=1)
        self.assertLess(objective.measure(compact, 512, 512), objective.measure(large, 512, 512))
        with patch("zae.losses.torch.randn_like", return_value=direction):
            objective(z, 512, 512).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertLess((z.grad * direction).sum(), 0)  # Descent moves toward the compact probe.

    def test_batch_mean_scales_each_sample_gradient(self):
        units = torch.randint(-20, 21, (1, 4, 8, 8)).float()
        direction = torch.randn_like(units)
        single = (units * self.scales).requires_grad_()
        pair = single.detach().repeat(2, 1, 1, 1).requires_grad_()
        objective = ZstdSizeLoss(self.scales, 127, self.cfg.zstd, directions=1)
        with patch("zae.losses.torch.randn_like", return_value=direction):
            objective(single, 128, 128).backward()
        with patch("zae.losses.torch.randn_like", return_value=direction.repeat(2, 1, 1, 1)):
            objective(pair, 128, 128).backward()
        self.assertGreater(single.grad.abs().sum(), 0)
        torch.testing.assert_close(pair.grad[0], single.grad[0] / 2, rtol=0, atol=0)
        torch.testing.assert_close(pair.grad[1], single.grad[0] / 2, rtol=0, atol=0)

    def test_penalty_changes_encoder_gradient_but_preserves_decoder_and_reconstruction(self):
        model = AutoEncoder(hidden_channels=8, latent_channels=4, downsampling_stages=1)
        reference = copy.deepcopy(model)
        x = torch.rand(1, 3, 32, 32)
        output, z = torch.compile(model, backend="eager", fullgraph=True)(x)
        expected, _ = reference(x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        F.mse_loss(expected, x).backward()
        (F.mse_loss(output, x) + 0.001 * self.objective(z, 32, 32)).backward()
        for a, b in zip(model.decoder.parameters(), reference.decoder.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a.grad, b.grad)
                            for a, b in zip(model.encoder.parameters(), reference.encoder.parameters())))
        with torch.no_grad():
            actual, _ = model(x)
        torch.testing.assert_close(actual, output, rtol=0, atol=0)

    def test_zero_weight_preserves_mse_gradients_and_rng(self):
        model = AutoEncoder(hidden_channels=8, latent_channels=4, downsampling_stages=1)
        reference = copy.deepcopy(model)
        x = torch.rand(1, 3, 32, 32)
        output, z = model(x)
        before = torch.get_rng_state().clone()
        loss = F.mse_loss(output, x) + 0 * self.objective(z.detach(), 32, 32)
        loss.backward()
        F.mse_loss(reference(x)[0], x).backward()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_bfloat16_forward_loss_is_float32(self):
        model = AutoEncoder(hidden_channels=8, latent_channels=4, downsampling_stages=1)
        x = torch.rand(1, 3, 32, 32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, z = model(x)
        rate = self.objective(z, 32, 32)
        loss = F.mse_loss(output.float(), x) + 0.001 * rate
        self.assertEqual(z.dtype, torch.bfloat16)
        self.assertEqual(rate.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
