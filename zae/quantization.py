"""Shared validation for frozen, training-only quantization calibration artifacts."""
import json
from pathlib import Path

import torch


def calibration_scales(calibration, cfg, data_manifest, device):
    expected = (cfg.quantization.scheme, cfg.quantization.dtype,
                cfg.quantization.qmax, data_manifest["dataset_id"])
    actual = tuple(calibration.get(k) for k in ("scheme", "dtype", "qmax", "dataset_id"))
    if actual != expected:
        raise ValueError("Calibration scheme, dtype, qmax or dataset does not match")
    scales = torch.tensor(calibration["scales"], dtype=torch.float32, device=device)
    if (scales.shape != (cfg.model.latent_channels,) or not torch.isfinite(scales).all()
            or (scales <= 0).any()):
        raise ValueError("Invalid calibration scales")
    return scales.reshape(1, -1, 1, 1)


def load_calibration(path, cfg, data_manifest, device):
    calibration = json.loads(Path(path).expanduser().read_text())
    return calibration_scales(calibration, cfg, data_manifest, device), calibration
