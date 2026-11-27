"""JPEG rate-distortion reference on the same images and metrics as evaluate.py.

    python jpeg_reference.py                       # 100 test images, all configured qualities
    python jpeg_reference.py runtime.device=cpu    # leave the GPU to a training job
    python jpeg_reference.py '+jpeg.compare={v1: results/baseline_5k/evaluation/summary.json}'

Sizes are complete JPEG files (headers included) from Pillow/libjpeg with fixed settings.
Quality is measured exactly like the autoencoder: decoded RGB8 against original RGB8.
"""
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
from PIL import Image

from zae.data import load_rgb_uint8, manifest, split_paths
from zae.metrics import psnr, ssim
from zae.tracking import (choose_device, log_context, seed_everything, setup_tracking,
                          validate_config, write_json)

log = logging.getLogger("jpeg_reference")


def encode_jpeg(array, quality, subsampling, optimize):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="JPEG", quality=quality, subsampling=subsampling,
                                optimize=optimize)
    return output.getvalue()


def decode(blob):
    with Image.open(io.BytesIO(blob)) as image:
        return np.array(image.convert("RGB"))


def to_tensor(array, device):
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)


def summarize(rows):
    result = []
    for quality in sorted({row["quality"] for row in rows}):
        selected = [row for row in rows if row["quality"] == quality]
        pixels = sum(row["height"] * row["width"] for row in selected)
        result.append({"quality": quality, "images": len(selected),
                       "bpp": 8 * sum(row["jpeg_bytes"] for row in selected) / pixels,
                       "psnr_mean": float(np.mean([row["psnr_db"] for row in selected])),
                       "ssim_mean": float(np.mean([row["ssim"] for row in selected]))})
    return result


def interpolate(curve, x_key, y_key, x):
    """Linear interpolation along the monotone JPEG curve; None outside the measured range."""
    points = sorted((point[x_key], point[y_key]) for point in curve)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    if not xs[0] <= x <= xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def compare(curve, summaries):
    """Place autoencoder summaries (evaluate.py summary.json, int8 branch) on the JPEG curve."""
    result = {}
    for label, path in summaries.items():
        branch = json.loads(Path(path).read_text())["int8"]
        bpp, quality = branch["bpp_after_zstd"], branch["psnr_mean"]
        jpeg_psnr = interpolate(curve, "bpp", "psnr_mean", bpp)
        jpeg_bpp = interpolate(curve, "psnr_mean", "bpp", quality)
        result[label] = {"source": str(path), "bpp": bpp, "psnr_mean": quality, "ssim_mean": branch["ssim_mean"],
                         "jpeg_psnr_at_same_bpp": jpeg_psnr,
                         "psnr_gap_db": None if jpeg_psnr is None else quality - jpeg_psnr,
                         "jpeg_bpp_at_same_psnr": jpeg_bpp,
                         "bits_ratio_vs_jpeg": None if jpeg_bpp is None else bpp / jpeg_bpp}
    return result


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
@torch.inference_mode()
def main(cfg):
    validate_config(cfg)
    seed_everything(cfg.experiment.seed, cfg.runtime.cpu_threads)
    device = choose_device(cfg.runtime.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    data_manifest = manifest(cfg)
    files = split_paths(cfg, data_manifest, cfg.evaluation.split)
    if cfg.evaluation.max_images:
        files = files[:cfg.evaluation.max_images]
    qualities = list(cfg.jpeg.qualities)
    setup_tracking(cfg)
    with mlflow.start_run(run_name=cfg.experiment.name + "_jpeg_reference"):
        mlflow.set_tags({"evaluation_split": cfg.evaluation.split, "codec": "jpeg",
                         "pillow_version": Image.__version__})
        log_context(cfg, out, data_manifest, "jpeg_reference", device)
        log.info("JPEG reference: %d %s images × qualities %s, subsampling %s, optimize %s, device %s",
                 len(files), cfg.evaluation.split, qualities, cfg.jpeg.subsampling, cfg.jpeg.optimize, device)
        rows, started = [], time.perf_counter()
        for number, path in enumerate(files, 1):
            array = load_rgb_uint8(path).permute(1, 2, 0).numpy()
            reference = to_tensor(array, device)
            h, w = array.shape[:2]
            parts = []
            for quality in qualities:
                blob = encode_jpeg(array, quality, cfg.jpeg.subsampling, cfg.jpeg.optimize)
                decoded = to_tensor(decode(blob), device)
                row = {"image": path.name, "quality": quality, "height": h, "width": w,
                       "jpeg_bytes": len(blob), "bpp": 8 * len(blob) / (h * w),
                       "psnr_db": psnr(decoded, reference).item(), "ssim": ssim(decoded, reference).item()}
                rows.append(row)
                parts.append(f"q{quality} {row['bpp']:.2f}bpp/{row['psnr_db']:.1f}dB")
            elapsed = time.perf_counter() - started
            log.info("[%d/%d] %s %d×%d | %s | ETA %ds", number, len(files), path.name, w, h, " ".join(parts),
                     elapsed / number * (len(files) - number))
        with (out / "jpeg_per_image.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        curve = summarize(rows)
        comparisons = compare(curve, {k: v for k, v in cfg.jpeg.compare.items()})
        write_json(out / "jpeg_summary.json", {"curve": curve, "comparisons": comparisons,
                                               "subsampling": cfg.jpeg.subsampling, "optimize": cfg.jpeg.optimize,
                                               "pillow_version": Image.__version__})
        for point in curve:
            mlflow.log_metrics({"jpeg_bpp": point["bpp"], "jpeg_psnr_mean": point["psnr_mean"],
                                "jpeg_ssim_mean": point["ssim_mean"]}, step=point["quality"])
        rule = "=" * 88
        log.info(rule)
        log.info("%8s %10s %10s %10s", "quality", "bpp", "PSNR dB", "SSIM")
        for point in curve:
            log.info("%8d %10.4f %10.3f %10.5f", point["quality"], point["bpp"], point["psnr_mean"], point["ssim_mean"])
        for label, item in comparisons.items():
            log.info("%s: %.3f bpp, %.3f dB | JPEG at the same bpp: %s | JPEG needs %s for the same PSNR", label,
                     item["bpp"], item["psnr_mean"],
                     "out of range" if item["jpeg_psnr_at_same_bpp"] is None else
                     f"{item['jpeg_psnr_at_same_bpp']:.3f} dB (gap {item['psnr_gap_db']:+.3f} dB)",
                     "out of range" if item["jpeg_bpp_at_same_psnr"] is None else
                     f"{item['jpeg_bpp_at_same_psnr']:.3f} bpp (autoencoder uses {item['bits_ratio_vs_jpeg']:.2f}×)")
        log.info(rule)
        for name in ["jpeg_per_image.csv", "jpeg_summary.json"]:
            mlflow.log_artifact(str(out / name))
        if cfg.tracking.plots:
            try:
                from zae import plots
                figure = plots.jpeg_reference(curve, comparisons, out / "plots" / "jpeg_reference.png",
                                              f"JPEG reference — {cfg.evaluation.split} split, {len(files)} images")
                mlflow.log_artifact(str(figure), "plots")
                log.info("Plot: %s", figure)
            except Exception:  # noqa: BLE001 - measurements are already saved
                log.exception("Plotting failed; measurements are unaffected")
        log_file = out / f"{HydraConfig.get().job.name}.log"
        if log_file.is_file():
            mlflow.log_artifact(str(log_file), "logs")
        print(json.dumps({"output_dir": str(out), "run_id": mlflow.active_run().info.run_id}), flush=True)


if __name__ == "__main__":
    main()
