"""Evaluate stored latent files, with and without quantization and Zstandard."""
import csv
import io
import json
from pathlib import Path

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image

from zae.codec import deserialize, latent_tensor, quantize, serialize
from zae.data import center_crop, load_rgb, manifest, split_paths
from zae.metrics import encode_padded, psnr, round_rgb, ssim
from zae.model import AutoEncoder
from zae.tracking import (choose_device, log_context, seed_everything, setup_tracking,
                          sha256, validate_config, write_json)


def png_bytes(x, level):
    array = x.squeeze(0).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG", compress_level=level)
    return output.getvalue()


@torch.inference_mode()
def calibrate(cfg, model, data_manifest, device, checkpoint_sha):
    qcfg = cfg.quantization
    if qcfg.calibration_file:
        calibration = json.loads(Path(qcfg.calibration_file).read_text())
        expected = ("uniform_symmetric_per_channel", qcfg.qmax, data_manifest["dataset_id"])
        actual = (calibration["scheme"], calibration["qmax"], calibration["dataset_id"])
        if actual != expected:
            raise ValueError("Calibration scheme, qmax or dataset does not match")
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
    scales = torch.tensor(calibration["scales"], dtype=torch.float32, device=device)
    if scales.shape != (cfg.model.latent_channels,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid calibration scales")
    return scales.reshape(1, -1, 1, 1), calibration


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
                       "serialized_bytes", "zstd_file_bytes", "reconstruction_png_bytes"]:
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
    with mlflow.start_run(run_name=cfg.experiment.name + "_evaluation"):
        mlflow.set_tags({"source_run_id": source_run, "checkpoint_sha256": checkpoint_sha,
                         "evaluation_split": cfg.evaluation.split,
                         "image_scope": "full_resolution" if cfg.evaluation.full_resolution else "center_crop"})
        log_context(cfg, out, data_manifest, "evaluation", device)
        scales = None
        if cfg.quantization.enabled:
            scales, calibration = calibrate(cfg, model, data_manifest, device, checkpoint_sha)
            write_json(out / "quantization.json", calibration)
            mlflow.log_artifact(str(out / "quantization.json"))
        files = split_paths(cfg, data_manifest, cfg.evaluation.split)
        if cfg.evaluation.max_images:
            files = files[:cfg.evaluation.max_images]
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
            branches = [("float32", z[0].cpu().numpy(), None, 0.0)]
            if scales is not None:
                q = quantize(z, scales, cfg.quantization.qmax)
                clipped = ((z / scales).abs() > cfg.quantization.qmax).float().mean().item()
                branches.append(("int8", q[0].cpu().numpy(), scales.flatten().tolist(), clipped))
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
                rows.append(row)
            print(f"evaluated {image_number}/{len(files)}: {path.name}", flush=True)
        with (out / "per_image.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows)
        write_json(out / "summary.json", summary)
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
        print(json.dumps({"output_dir": str(out), "run_id": mlflow.active_run().info.run_id,
                          "source_run_id": source_run}), flush=True)


if __name__ == "__main__":
    main()
