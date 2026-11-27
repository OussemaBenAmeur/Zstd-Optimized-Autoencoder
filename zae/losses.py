"""Actual Zstd file-size loss with a two-point Gaussian gradient estimator.

Forward measures files with the existing codec. Backward estimates the gradient
of Gaussian-smoothed size using actual compressed sizes of paired perturbations.
The model forward pass and decoder inputs are never modified.
"""
import json
import math
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from zae.codec import serialize


class _ZstdBpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, objective, height, width):
        units = z.detach().to(device="cpu", dtype=torch.float32) / objective.scales
        ctx.save_for_backward(units)
        ctx.objective, ctx.height, ctx.width = objective, height, width
        return objective.measure(units, height, width).to(z.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (units,) = ctx.saved_tensors
        objective = ctx.objective
        gradient = torch.zeros_like(units)
        sigma = objective.perturbation_symbols
        for _ in range(objective.directions):
            # CPU torch RNG is already seeded/checkpointed by the v3 pipeline.
            direction = torch.randn_like(units)
            plus = objective.measure(units + sigma * direction, ctx.height, ctx.width)
            minus = objective.measure(units - sigma * direction, ctx.height, ctx.width)
            slope = (plus - minus) / (2 * sigma * objective.directions)
            gradient.add_(slope[:, None, None, None] * direction)
        # u = z / scales. grad_output carries the batch reduction and loss weight.
        gradient.div_(objective.scales)
        return gradient.to(grad_output.device) * grad_output[:, None, None, None], None, None, None


class ZstdSizeLoss:
    """Per-image int8 + Zstd file bpp with finite-difference gradients in symbol units."""
    def __init__(self, scales, qmax, zstd_config, perturbation_symbols=1.0, directions=4):
        self.scales = torch.as_tensor(scales, dtype=torch.float32, device="cpu").detach().clone().reshape(1, -1, 1, 1)
        if not torch.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("Loss calibration scales must be finite and positive")
        if isinstance(qmax, bool) or not isinstance(qmax, int) or not 1 <= qmax <= 127:
            raise ValueError("Loss qmax must be an integer between 1 and 127")
        if (isinstance(perturbation_symbols, bool) or not isinstance(perturbation_symbols, (int, float))
                or not math.isfinite(perturbation_symbols) or perturbation_symbols <= 0):
            raise ValueError("perturbation_symbols must be finite and positive")
        if isinstance(directions, bool) or not isinstance(directions, int) or directions < 1:
            raise ValueError("directions must be a positive integer")
        self.qmax, self.zstd_config = qmax, zstd_config
        self.perturbation_symbols, self.directions = perturbation_symbols, directions

    @classmethod
    def from_config(cls, cfg, data_manifest):
        loss_cfg = cfg.training.zstd_loss
        calibration = json.loads(Path(loss_cfg.calibration_file).expanduser().read_text())
        expected = ("uniform_symmetric_per_channel", "int8", loss_cfg.qmax,
                    data_manifest["dataset_id"])
        actual = tuple(calibration.get(k) for k in ("scheme", "dtype", "qmax", "dataset_id"))
        if actual != expected or len(calibration["scales"]) != cfg.model.latent_channels:
            raise ValueError("Loss calibration does not match the model, quantizer or dataset")
        return cls(calibration["scales"], loss_cfg.qmax, loss_cfg.codec,
                   loss_cfg.perturbation_symbols, loss_cfg.directions)

    def measure(self, units, height, width):
        """Exact complete file sizes, normalized to bits per original input pixel."""
        arrays = units.round().clamp(-self.qmax, self.qmax).to(torch.int8).numpy()
        metadata = {"image_height": height, "image_width": width, "quantized": True,
                    "scales": self.scales.flatten().tolist(), "checkpoint_sha256": "0" * 64}
        # Fixed-width checkpoint hashes are in the uncompressed header, so the
        # placeholder gives the same size as an evaluation file with a real hash.
        return torch.tensor([8 * len(serialize(array, metadata, self.zstd_config)) / (height * width)
                             for array in arrays], dtype=torch.float32)

    def __call__(self, z, height, width):
        if z.ndim != 4 or z.shape[1] != self.scales.shape[1]:
            raise ValueError("Expected NCHW latents matching the loss calibration")
        return _ZstdBpp.apply(z.float(), self, height, width).mean()
