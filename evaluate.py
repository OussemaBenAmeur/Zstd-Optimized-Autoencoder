"""Evaluate stored latent files, with and without quantization and Zstandard."""
import csv
import io
import json
import logging
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image

from zae.codec import deserialize, latent_tensor, quantize, serialize
from zae.data import center_crop, load_rgb, manifest, split_paths
from zae.metrics import encode_padded, psnr, round_rgb, ssim
from zae.model import AutoEncoder
from zae.quantization import calibration_scales, load_calibration
from zae.tracking import (choose_device, log_context, seed_everything, setup_tracking,
                          sha256, validate_config, write_json)

log = logging.getLogger("evaluate")


def byte_entropy(payload):
    """Order-0 entropy (bits/byte) of the raw latent bytes: the floor for a byte-wise entropy coder."""
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def entropy_bits(counts):
    counts = np.asarray(counts, dtype=np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def to_image(x):
    return x[0].permute(1, 2, 0).float().cpu().numpy()


def png_bytes(x, level):
    array = x.squeeze(0).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG", compress_level=level)
    return output.getvalue()


@torch.inference_mode()
def calibrate(cfg, model, data_manifest, device, checkpoint_sha, checkpoint=None):
    qcfg = cfg.quantization
    if checkpoint is not None and checkpoint.get("config", {}).get("training", {}).get("loss") == "mse_zstd":
        calibration = checkpoint.get("quantization_calibration")
        if calibration is None:
            raise ValueError("Zstd-trained checkpoint is missing its frozen quantizer")
        scales = calibration_scales(calibration, cfg, data_manifest, device)
        if qcfg.calibration_file:
            external_scales, _ = load_calibration(qcfg.calibration_file, cfg, data_manifest, device)
            if not torch.equal(scales, external_scales):
                raise ValueError("Evaluation quantizer differs from the frozen training quantizer")
        return scales, calibration
    if qcfg.calibration_file:
        return load_calibration(qcfg.calibration_file, cfg, data_manifest, device)
    else:
        files = split_paths(cfg, data_manifest, "train")[:qcfg.calibration_images]
        values = []
        for path in files:
            x = center_crop(load_rgb(path), qcfg.calibration_crop_size).unsqueeze(0).to(device)
            z = encode_padded(model, x)
            values.append(z[0].flatten(1).abs().cpu())
        scales = (torch.quantile(torch.cat(values, 1), qcfg.calibration_percentile / 100, dim=1)
                  / qcfg.qmax).clamp_min(1e-8)
        calibration = {
            "scheme": qcfg.scheme, "dtype": qcfg.dtype, "qmax": qcfg.qmax,
            "scales": scales.tolist(), "dataset_id": data_manifest["dataset_id"],
            "source_checkpoint_sha256": checkpoint_sha,
            "percentile": qcfg.calibration_percentile,
            "crop_size": qcfg.calibration_crop_size,
            "images": [p.relative_to(Path(cfg.data.root).expanduser().resolve()).as_posix() for p in files],
        }
    return calibration_scales(calibration, cfg, data_manifest, device), calibration


def summarize(rows):
    result = {}
    for branch in sorted({row["branch"] for row in rows}):
        selected = [row for row in rows if row["branch"] == branch]
        total_pixels = sum(row["height"] * row["width"] for row in selected)
        values = {
            "images": len(selected),
            "psnr_mean": sum(row["psnr_db"] for row in selected) / len(selected),
            "ssim_mean": sum(row["ssim"] for row in selected) / len(selected),
        }
        for column in ["original_png_bytes", "raw_rgb_bytes", "latent_payload_bytes",
                       "serialized_bytes", "zstd_file_bytes", "reconstruction_png_bytes",
                       "order0_entropy_bytes"]:
            if column not in selected[0]:
                continue
            total = sum(row[column] for row in selected)
            values[column + "_total"] = total
            values[column + "_mean"] = total / len(selected)
        values["bpp_before_zstd"] = 8 * values["serialized_bytes_total"] / total_pixels
        values["bpp_after_zstd"] = 8 * values["zstd_file_bytes_total"] / total_pixels
        values["compression_ratio_png_after_zstd"] = values["original_png_bytes_total"] / values["zstd_file_bytes_total"]
        values["compression_ratio_rgb_after_zstd"] = values["raw_rgb_bytes_total"] / values["zstd_file_bytes_total"]
        result[branch] = values
    return result


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
@torch.inference_mode()
def main(cfg):
    validate_config(cfg)
    if not cfg.evaluation.checkpoint:
        raise ValueError("Specify evaluation.checkpoint=/path/to/best.pt")
    seed_everything(cfg.experiment.seed, cfg.runtime.cpu_threads)
    device = choose_device(cfg.runtime.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    data_manifest = manifest(cfg)
    checkpoint = torch.load(cfg.evaluation.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["model_config"] != OmegaConf.to_container(cfg.model):
        raise ValueError("Model config differs from checkpoint; use the same config as training")
    if checkpoint["manifest"] != data_manifest:
        raise ValueError("Dataset contents or split differs from the checkpoint")
    model = AutoEncoder(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    checkpoint_sha = sha256(cfg.evaluation.checkpoint)
    source_run = cfg.evaluation.source_run_id or checkpoint["run_id"]
    setup_tracking(cfg)
    with mlflow.start_run(run_name=cfg.experiment.name + "_evaluation",
                          log_system_metrics=cfg.tracking.system_metrics):
        mlflow.set_tags({"source_run_id": source_run, "checkpoint_sha256": checkpoint_sha,
                         "evaluation_split": cfg.evaluation.split,
                         "image_scope": "full_resolution" if cfg.evaluation.full_resolution else "center_crop"})
        log_context(cfg, out, data_manifest, "evaluation", device)
        scales = None
        if cfg.quantization.enabled:
            scales, calibration = calibrate(cfg, model, data_manifest, device, checkpoint_sha, checkpoint)
            values = calibration["scales"]
            log.info("Quantization: %s, qmax %d, scales %s (min %.4g, median %.4g, max %.4g)", calibration["scheme"],
                     calibration["qmax"], "loaded from " + str(cfg.quantization.calibration_file)
                     if cfg.quantization.calibration_file else
                     f"calibrated at p{calibration['percentile']} on {len(calibration['images'])} training crops",
                     min(values), float(np.median(values)), max(values))
            write_json(out / "quantization.json", calibration)
            mlflow.log_artifact(str(out / "quantization.json"))
        files = split_paths(cfg, data_manifest, cfg.evaluation.split)
        if cfg.evaluation.max_images:
            files = files[:cfg.evaluation.max_images]
        log.info("Evaluating %d %s images (%s) with checkpoint %s (sha256 %s…, step %s, training run %s)",
                 len(files), cfg.evaluation.split,
                 "full resolution" if cfg.evaluation.full_resolution else f"{cfg.validation.crop_size}² center crops",
                 cfg.evaluation.checkpoint, checkpoint_sha[:12], checkpoint.get("step"), source_run)
        log.info("Zstd level %d, checksum %s, threads %d; PNG level %d", cfg.zstd.level, cfg.zstd.checksum,
                 cfg.zstd.threads, cfg.evaluation.png_compress_level)
        channels = cfg.model.latent_channels
        symbol_counts = np.zeros(256, dtype=np.int64)
        channel_counts = np.zeros((channels, 256), dtype=np.int64)
        channel_clipped = np.zeros(channels, dtype=np.int64)
        channel_values = 0
        float_sum = np.zeros(channels)
        float_square_sum = np.zeros(channels)
        float_abs_max = np.zeros(channels)
        samples, started = [], time.perf_counter()
        rows = []
        for image_number, path in enumerate(files, 1):
            x = load_rgb(path).unsqueeze(0).to(device)
            source_h, source_w = x.shape[-2:]
            if not cfg.evaluation.full_resolution:
                x = center_crop(x[0], cfg.validation.crop_size).unsqueeze(0)
            h, w = x.shape[-2:]
            original_bytes = path.stat().st_size if cfg.evaluation.full_resolution else len(
                png_bytes(x, cfg.evaluation.png_compress_level))
            z = encode_padded(model, x)
            flat = z[0].flatten(1).double()
            float_sum += flat.sum(1).cpu().numpy()
            float_square_sum += flat.square().sum(1).cpu().numpy()
            float_abs_max = np.maximum(float_abs_max, flat.abs().max(1).values.cpu().numpy())
            branches = [("float32", z[0].cpu().numpy(), None, 0.0)]
            if scales is not None:
                q = quantize(z, scales, cfg.quantization.qmax)
                over = (z / scales).abs() > cfg.quantization.qmax
                clipped = over.float().mean().item()
                channel_clipped += over[0].flatten(1).sum(1).cpu().numpy()
                channel_values += over[0, 0].numel()
                symbols = q[0].flatten(1).cpu().numpy().astype(np.int64) + 128
                for channel in range(channels):
                    channel_counts[channel] += np.bincount(symbols[channel], minlength=256)
                symbol_counts += np.bincount(symbols.ravel(), minlength=256)
                branches.append(("int8", q[0].cpu().numpy(), scales.flatten().tolist(), clipped))
            sample = {"name": path.name, "reconstructions": {}} if len(samples) < cfg.evaluation.plot_samples else None
            image_log = []
            for branch, array, channel_scales, clipped in branches:
                metadata = {
                    "image_height": h, "image_width": w, "quantized": branch == "int8",
                    "scales": channel_scales, "checkpoint_sha256": checkpoint_sha,
                }
                stored = serialize(array, metadata)
                compressed = serialize(array, metadata, cfg.zstd)
                plain_array, plain_info = deserialize(stored)
                restored, info = deserialize(compressed)
                if restored.tobytes() != plain_array.tobytes() or info != plain_info:
                    raise RuntimeError("Zstandard did not restore identical latent bytes/metadata")
                decoded_z = latent_tensor(restored, info, device)
                reconstruction = round_rgb(model.decode(decoded_z)[..., :h, :w])
                reconstruction_bytes = png_bytes(reconstruction, cfg.evaluation.png_compress_level)
                name = f"{path.stem}_{branch}"
                if cfg.evaluation.save_latent_files:
                    directory = out / "latents"
                    directory.mkdir(exist_ok=True)
                    (directory / (name + ".zae")).write_bytes(stored)
                    (directory / (name + ".zae.zst")).write_bytes(compressed)
                if cfg.evaluation.save_reconstructions:
                    directory = out / "reconstructions"
                    directory.mkdir(exist_ok=True)
                    (directory / (name + ".png")).write_bytes(reconstruction_bytes)
                row = {
                    "image": path.name, "branch": branch,
                    "height": h, "width": w, "source_height": source_h, "source_width": source_w,
                    "is_crop": not cfg.evaluation.full_resolution,
                    "source_png_bytes": path.stat().st_size, "original_png_bytes": original_bytes,
                    "raw_rgb_bytes": h * w * 3, "latent_shape": json.dumps(list(array.shape)),
                    "latent_dtype": array.dtype.str, "latent_payload_bytes": array.nbytes,
                    "serialized_bytes": len(stored), "zstd_file_bytes": len(compressed),
                    "reconstruction_png_bytes": len(reconstruction_bytes),
                    "bpp_before_zstd": 8 * len(stored) / (h * w),
                    "bpp_after_zstd": 8 * len(compressed) / (h * w),
                    "compression_ratio_png_before_zstd": original_bytes / len(stored),
                    "compression_ratio_png_after_zstd": original_bytes / len(compressed),
                    "compression_ratio_rgb_before_zstd": (h * w * 3) / len(stored),
                    "compression_ratio_rgb_after_zstd": (h * w * 3) / len(compressed),
                    "zstd_size_reduction_factor": len(stored) / len(compressed),
                    "psnr_db": psnr(reconstruction, x).item(), "ssim": ssim(reconstruction, x).item(),
                    "quantization_clipped_fraction": clipped, "zstd_roundtrip_exact": True,
                }
                entropy = byte_entropy(array.tobytes(order="C"))
                row["payload_byte_entropy_bits"] = entropy
                row["order0_entropy_bytes"] = entropy * array.nbytes / 8
                row["zstd_bytes_over_order0"] = len(compressed) / max(row["order0_entropy_bytes"], 1e-9)
                rows.append(row)
                image_log.append(f"{branch} {row['psnr_db']:.2f} dB / SSIM {row['ssim']:.4f} | "
                                 f"{len(stored) / 1024:.1f}→{len(compressed) / 1024:.1f} KiB Zstd "
                                 f"({row['bpp_after_zstd']:.3f} bpp, ×{row['zstd_size_reduction_factor']:.3f}, "
                                 f"H0 {entropy:.2f} bit/B)" + (f" clip {100 * clipped:.3f}%" if branch == "int8" else ""))
                if sample is not None:
                    crop = min(256, h, w)
                    sample["reconstructions"][branch] = {
                        "image": to_image(center_crop(reconstruction[0], crop).unsqueeze(0)),
                        "psnr": row["psnr_db"], "bpp": row["bpp_after_zstd"]}
            if sample is not None:
                sample["original"] = to_image(center_crop(x[0], min(256, h, w)).unsqueeze(0))
                samples.append(sample)
            elapsed = time.perf_counter() - started
            log.info("[%d/%d] %s %d×%d | %s | ETA %ds", image_number, len(files), path.name, w, h,
                     " || ".join(image_log), elapsed / image_number * (len(files) - image_number))
        with (out / "per_image.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows)
        write_json(out / "summary.json", summary)
        pixels = channel_values or 1
        stats = None
        float_count = sum(r["latent_payload_bytes"] // 4 for r in rows if r["branch"] == "float32") / channels
        float_mean = float_sum / max(float_count, 1)
        latent_stats = {"float32_channel_mean": float_mean.tolist(),
                        "float32_channel_std": np.sqrt(np.maximum(
                            float_square_sum / max(float_count, 1) - float_mean ** 2, 0)).tolist(),
                        "float32_channel_abs_max": float_abs_max.tolist()}
        if scales is not None:
            latent_stats.update({
                "scales": scales.flatten().tolist(), "qmax": cfg.quantization.qmax,
                "int8_symbol_counts": symbol_counts.tolist(), "int8_entropy_bits": entropy_bits(symbol_counts),
                "int8_channel_entropy_bits": [entropy_bits(c) for c in channel_counts],
                "int8_channel_clipped_fraction": (channel_clipped / pixels).tolist(),
                "int8_zero_fraction": float(symbol_counts[128] / symbol_counts.sum()),
            })
            stats = latent_stats
        write_json(out / "latent_stats.json", latent_stats)
        mlflow.log_artifact(str(out / "latent_stats.json"))
        rule = "=" * 88
        log.info(rule)
        log.info("%-8s %6s %9s %8s %12s %12s %9s %9s %9s", "branch", "images", "PSNR dB", "SSIM",
                 "file bpp", "Zstd bpp", "Zstd ×", "vs PNG ×", "H0 bpp")
        for branch, values in summary.items():
            log.info("%-8s %6d %9.3f %8.5f %12.4f %12.4f %9.4f %9.3f %9.4f", branch, values["images"],
                     values["psnr_mean"], values["ssim_mean"], values["bpp_before_zstd"], values["bpp_after_zstd"],
                     values["serialized_bytes_total"] / values["zstd_file_bytes_total"],
                     values["compression_ratio_png_after_zstd"],
                     8 * values.get("order0_entropy_bytes_total", float("nan")) / (
                         values["raw_rgb_bytes_total"] / 3))
        if stats:
            log.info("int8 symbols: pooled entropy %.3f bits, %.2f%% zeros, mean clipped %.4f%%",
                     stats["int8_entropy_bits"], 100 * stats["int8_zero_fraction"],
                     100 * float(np.mean(stats["int8_channel_clipped_fraction"])))
        log.info(rule)
        if cfg.tracking.plots:
            try:
                from zae import plots
                title = f"{cfg.experiment.name} — {cfg.evaluation.split} split, checkpoint {checkpoint_sha[:10]}"
                figures = plots.evaluation_figures(rows, summary, stats, out / "plots", title)
                if samples:
                    figures.append(plots.evaluation_samples(samples, out / "plots" / "samples.png",
                                                            f"Center crops — {cfg.evaluation.split} split"))
                for figure in figures:
                    mlflow.log_artifact(str(figure), "plots")
                log.info("Plots: %s", out / "plots")
            except Exception:  # noqa: BLE001 - measurements are already saved
                log.exception("Plotting failed; measurements are unaffected")
        for branch, values in summary.items():
            mlflow.log_metrics({f"{branch}.{key}": value for key, value in values.items()})
        mlflow.log_metrics({
            "model_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "decoder_weight_bytes": sum(p.numel() * p.element_size() for p in model.decoder.parameters()),
            "checkpoint_file_bytes": Path(cfg.evaluation.checkpoint).stat().st_size,
        })
        for name in ["per_image.csv", "summary.json"]:
            mlflow.log_artifact(str(out / name))
        for name in ["latents", "reconstructions"]:
            if (out / name).is_dir():
                mlflow.log_artifacts(str(out / name), name)
        log_file = out / f"{HydraConfig.get().job.name}.log"
        if log_file.is_file():
            mlflow.log_artifact(str(log_file), "logs")
        print(json.dumps({"output_dir": str(out), "run_id": mlflow.active_run().info.run_id,
                          "source_run_id": source_run}), flush=True)


if __name__ == "__main__":
    main()
