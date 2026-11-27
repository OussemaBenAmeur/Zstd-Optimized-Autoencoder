"""Train reconstruction or Zstd-oriented models with the same Hydra/MLflow pipeline."""
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from zae.data import TrainingCrops, center_crop, load_rgb, manifest, split_paths
from zae.early_stopping import EarlyStopping
from zae.losses import ZstdRateLoss, zstd_file_sizes
from zae.metrics import encode_padded, psnr, round_rgb, ssim
from zae.model import AutoEncoder
from zae.quantization import load_calibration
from zae.tracking import (choose_device, log_context, seed_everything,
                          setup_tracking, validate_config, write_json)

log = logging.getLogger("train")


def learning_rate(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    # Reaches the configured minimum on the last update.
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps - 1)
    return cfg.minimum_learning_rate + 0.5 * (cfg.learning_rate - cfg.minimum_learning_rate) * (
        1 + math.cos(math.pi * min(progress, 1)))


@torch.inference_mode()
def validate(model, images, batch_size, device, objective=None, zstd_config=None):
    model.eval()
    psnrs, ssims, latents = [], [], []
    mse_sum, file_bytes, pixels = 0.0, 0, 0
    clipped, symbols = 0, 0
    for start in range(0, len(images), batch_size):
        x = torch.stack(images[start:start + batch_size]).to(device)
        h, w = x.shape[-2:]
        z = encode_padded(model, x)
        decoded_z = objective.quantized_latent(z) if objective is not None else z
        decoded = model.decode(decoded_z)[..., :h, :w]
        reconstruction = round_rgb(decoded)
        if objective is not None:
            mse_sum += F.mse_loss(decoded, x, reduction="none").flatten(1).mean(1).sum().item()
            file_bytes += sum(zstd_file_sizes(z, objective.scales, objective.qmax, h, w, zstd_config))
            pixels += x.shape[0] * h * w
            clipped += ((z / objective.scales).abs() > objective.qmax).sum().item()
            symbols += z.numel()
        psnrs.extend(psnr(reconstruction, x).tolist())
        ssims.extend(ssim(reconstruction, x).tolist())
        latents.append(z.flatten().float().cpu())
    model.train()
    z = torch.cat(latents)
    magnitude = z.abs()
    # Latent scale matters for the later quantization/Zstd study, so it is tracked from the start.
    sample = magnitude if magnitude.numel() <= 16_000_000 else magnitude[::magnitude.numel() // 16_000_000 + 1]
    record = {"val_psnr": sum(psnrs) / len(psnrs), "val_ssim": sum(ssims) / len(ssims),
            "val_psnr_min": min(psnrs), "val_psnr_max": max(psnrs),
            "val_psnr_std": float(np.std(psnrs)), "val_ssim_min": min(ssims),
            "latent_mean": z.mean().item(), "latent_std": z.std().item(),
            "latent_abs_p999": torch.quantile(sample, 0.999).item(), "latent_abs_max": magnitude.max().item()}
    if objective is not None:
        record.update({"val_mse": mse_sum / len(images), "val_zstd_bpp": 8 * file_bytes / pixels,
                       "val_zstd_file_bytes": file_bytes, "val_quantization_clipped_fraction": clipped / symbols})
        record["val_rate_distortion"] = record["val_mse"] + objective.lambda_rate * record["val_zstd_bpp"]
    return record


@torch.inference_mode()
def reconstruction_examples(model, images, names, device, objective=None):
    model.eval()
    examples = []
    for image, name in zip(images, names):
        x = image.unsqueeze(0).to(device)
        h, w = x.shape[-2:]
        z = encode_padded(model, x)
        if objective is not None:
            z = objective.quantized_latent(z)
        reconstruction = round_rgb(model.decode(z)[..., :h, :w])
        examples.append({"name": name, "psnr": psnr(reconstruction, x).item(), "ssim": ssim(reconstruction, x).item(),
                         "original": x[0].permute(1, 2, 0).cpu().numpy(),
                         "reconstruction": reconstruction[0].permute(1, 2, 0).cpu().numpy()})
    model.train()
    return examples


def save_checkpoint(path, model, optimizer, step, best, cfg, data_manifest, history,
                    early_stopping, training_log_state, performance=None, quantization_calibration=None):
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "best_psnr": best, "model_config": OmegaConf.to_container(cfg.model),
        "config": OmegaConf.to_container(cfg, resolve=True), "manifest": data_manifest,
        "history": history, "run_id": mlflow.active_run().info.run_id,
        "early_stopping": early_stopping.state_dict(),
        "training_log_state": training_log_state,
        # Wall-clock measurements are kept apart from the reproducible training history.
        "performance": performance or [],
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if quantization_calibration is not None:
        state["quantization_calibration"] = quantization_calibration
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def check_resume(checkpoint, cfg, data_manifest):
    saved = checkpoint["config"]
    # Paths and hardware can change between PCs; the experiment and data must agree.
    for section in ["model", "training", "validation"]:
        current = OmegaConf.to_container(cfg[section], resolve=True)
        previous = dict(saved[section])
        if section == "training":
            current.pop("resume_checkpoint", None)
            previous.pop("resume_checkpoint", None)
        if current != previous:
            raise ValueError(f"Cannot resume with changed {section} settings")
    for field in ["crop_size", "horizontal_flip"]:
        if cfg.data[field] != saved["data"][field]:
            raise ValueError(f"Cannot resume with changed data.{field}")
    if cfg.experiment.seed != saved["experiment"]["seed"]:
        raise ValueError("Cannot resume with changed seed")
    if data_manifest != checkpoint["manifest"]:
        raise ValueError("Dataset contents or split changed since checkpoint")
    if cfg.training.loss == "mse_zstd":
        for section in ("quantization", "zstd"):
            current = OmegaConf.to_container(cfg[section], resolve=True)
            previous = dict(saved[section])
            if section == "quantization":
                # Artifact location can change; its contents are checked below.
                current.pop("calibration_file", None)
                previous.pop("calibration_file", None)
            if current != previous:
                raise ValueError(f"Cannot resume with changed {section} settings")
        _, calibration = load_calibration(cfg.quantization.calibration_file, cfg, data_manifest, "cpu")
        if calibration != checkpoint.get("quantization_calibration"):
            raise ValueError("Frozen calibration changed since checkpoint")
    if checkpoint.get("early_stopping", {}).get("stopped", False):
        raise ValueError("Checkpoint has already early-stopped; start a new experiment to change the policy")


def clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def describe_run(cfg, out, device, model, data_manifest, start, checkpoint):
    t, v = cfg.training, cfg.validation
    rule = "=" * 88
    log.info(rule)
    log.info("Experiment '%s': %s", cfg.experiment.name, cfg.experiment.description)
    log.info("Output directory: %s", out)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        log.info("Device: %s (%s, %.1f GiB, compute capability %d.%d)", device, properties.name,
                 properties.total_memory / 2 ** 30, properties.major, properties.minor)
    else:
        log.info("Device: %s (%d CPU threads)", device, torch.get_num_threads())
    log.info("PyTorch %s | CUDA %s | cuDNN %s | float32 matmul precision '%s'", torch.__version__,
             torch.version.cuda, torch.backends.cudnn.version(), torch.get_float32_matmul_precision())
    log.info("Dataset %s… under %s: %d train / %d validation / %d test images",
             data_manifest["dataset_id"][:12], Path(cfg.data.root).expanduser().resolve(),
             len(data_manifest["train"]), len(data_manifest["validation"]), len(data_manifest["test"]))
    log.info("Validation images: %s", ", ".join(Path(p).name for p in data_manifest["validation"]))
    factor = model.factor
    channels = cfg.model.latent_channels
    log.info("Model: %s, %d hidden / %d latent channels, %d downsampling stages (factor %d)",
             cfg.model.name, cfg.model.hidden_channels, channels, cfg.model.downsampling_stages, factor)
    log.info("Parameters: %s total = %s encoder + %s decoder (%.2f MiB float32 decoder)",
             f"{parameter_count(model):,}", f"{parameter_count(model.encoder):,}",
             f"{parameter_count(model.decoder):,}", 4 * parameter_count(model.decoder) / 2 ** 20)
    crop = cfg.data.crop_size
    log.info("Latent for a %d² crop: %d×%d×%d = %.4f values/pixel -> %.2f bpp float32, %.2f bpp int8 "
             "before Zstd", crop, channels, crop // factor, crop // factor, channels / factor ** 2,
             32 * channels / factor ** 2, 8 * channels / factor ** 2)
    crops = t.max_steps * t.batch_size
    log.info("Budget: up to %s steps × batch %d = %s random %d² crops (≈%.0f visits per training image)",
             f"{t.max_steps:,}", t.batch_size, f"{crops:,}", crop, crops / len(data_manifest["train"]))
    log.info("Optimizer: Adam, lr %.2e, linear warmup %d steps, cosine to %.1e, gradient clip %.2f, loss %s",
             t.learning_rate, t.warmup_steps, t.minimum_learning_rate, t.gradient_clip, t.loss)
    es = t.early_stopping
    log.info("Validation: every %d steps on %d center crops of %d²; best checkpoint by %s; early stopping %s",
             v.every_steps, len(data_manifest["validation"]), v.crop_size, v.checkpoint_metric,
             f"on (patience {es.patience} checks, min gain {es.min_delta} dB)" if es.enabled else "off")
    log.info("Data loader: %d workers, horizontal flip %s; logging every %d steps",
             cfg.runtime.num_workers, cfg.data.horizontal_flip, cfg.tracking.log_every_steps)
    if checkpoint:
        log.info("RESUMING from %s at step %d (best val PSNR %.3f dB, patience %d/%d)",
                 cfg.training.resume_checkpoint, start, checkpoint["best_psnr"],
                 checkpoint["early_stopping"]["bad_checks"], es.patience)
    log.info("MLflow run %s at %s", mlflow.active_run().info.run_id, mlflow.get_tracking_uri())
    log.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True).rstrip())
    log.info(rule)


def write_model_summary(model, path):
    lines = [str(model), "", f"{'parameter':60s} {'shape':>22s} {'count':>12s}"]
    for name, parameter in model.named_parameters():
        lines.append(f"{name:60s} {str(tuple(parameter.shape)):>22s} {parameter.numel():>12,}")
    lines.append(f"{'total':60s} {'':>22s} {parameter_count(model):>12,}")
    Path(path).write_text("\n".join(lines) + "\n")


def refresh_plots(cfg, out, history, performance, examples, step, final=False):
    """Plot failures are reported but never interrupt a training run."""
    if not cfg.tracking.plots:
        return
    try:
        from zae import plots
        directory = out / "plots"
        title = f"{cfg.experiment.name} — step {step:,}" + (" (final)" if final else "")
        curves = plots.training_curves(history, performance, directory / "training_curves.png", title,
                                       OmegaConf.to_container(cfg.training.early_stopping),
                                       cfg.training.gradient_clip)
        mlflow.log_artifact(str(curves), "plots")
        if cfg.training.loss == "mse_zstd":
            rate_curves = plots.rate_training_curves(history, directory / "rate_training_curves.png", title)
            mlflow.log_artifact(str(rate_curves), "plots")
        if examples:
            grid = plots.reconstruction_grid(examples, directory / "reconstructions" / f"step_{step:06d}.png",
                                             f"Validation reconstructions — step {step:,}")
            mlflow.log_artifact(str(grid), "plots/reconstructions")
    except Exception:  # noqa: BLE001 - diagnostics must not cost a long run
        log.exception("Plotting failed at step %d; training continues", step)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    validate_config(cfg)
    seed_everything(cfg.experiment.seed, cfg.runtime.cpu_threads)
    device = choose_device(cfg.runtime.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    log.info("Hashing dataset contents and building the split manifest…")
    data_manifest = manifest(cfg)
    model = AutoEncoder(**OmegaConf.to_container(cfg.model)).to(device)
    objective, calibration = None, None
    if cfg.training.loss == "mse_zstd":
        scales, calibration = load_calibration(cfg.quantization.calibration_file, cfg, data_manifest, device)
        objective = ZstdRateLoss(scales, cfg.quantization.qmax,
                                 **OmegaConf.to_container(cfg.training.zstd_loss))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    early_stopping = EarlyStopping(**OmegaConf.to_container(cfg.training.early_stopping))
    start, best, history, checkpoint, performance = 0, -float("inf"), [], None, []
    if cfg.training.resume_checkpoint:
        checkpoint = torch.load(cfg.training.resume_checkpoint, map_location="cpu", weights_only=False)
        check_resume(checkpoint, cfg, data_manifest)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        start, best, history = checkpoint["step"], checkpoint["best_psnr"], checkpoint["history"]
        performance = checkpoint.get("performance", [])
        early_stopping.load_state_dict(checkpoint["early_stopping"])
        if start >= cfg.training.max_steps:
            raise ValueError("Checkpoint already reached max_steps; no training is needed")
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    validation_paths = split_paths(cfg, data_manifest, "validation")
    monitor = [center_crop(load_rgb(p), cfg.validation.crop_size) for p in validation_paths]
    sample_count = min(cfg.tracking.reconstruction_samples, len(monitor))
    sample_images, sample_names = monitor[:sample_count], [p.name for p in validation_paths[:sample_count]]
    dataset = TrainingCrops(split_paths(cfg, data_manifest, "train"), cfg.data.crop_size,
                            cfg.training.max_steps * cfg.training.batch_size,
                            cfg.experiment.seed, cfg.data.horizontal_flip)
    # DataLoader gets a separate generator so its worker setup does not perturb model RNG.
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size,
                        sampler=range(start * cfg.training.batch_size, len(dataset)),
                        num_workers=cfg.runtime.num_workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(cfg.experiment.seed))
    setup_tracking(cfg)
    run_args = {"run_id": checkpoint["run_id"]} if checkpoint else {"run_name": cfg.experiment.name}
    with mlflow.start_run(**run_args, log_system_metrics=cfg.tracking.system_metrics):
        log_context(cfg, out, data_manifest, "training", device)
        if calibration is not None:
            write_json(out / "quantization.json", calibration)
            mlflow.log_artifact(str(out / "quantization.json"))
            mlflow.set_tag("rate_gradient", "soft_histogram_and_chw_differences_proxy")
        describe_run(cfg, out, device, model, data_manifest, start, checkpoint)
        write_model_summary(model, out / "model_summary.txt")
        mlflow.log_artifact(str(out / "model_summary.txt"))
        mlflow.log_metric("model_parameters", parameter_count(model))
        mlflow.log_metric("encoder_parameters", parameter_count(model.encoder))
        mlflow.log_metric("decoder_parameters", parameter_count(model.decoder))
        mlflow.log_metric("model_weight_bytes", sum(p.numel() * p.element_size() for p in model.parameters()))
        mlflow.log_metric("decoder_weight_bytes", sum(p.numel() * p.element_size() for p in model.decoder.parameters()))
        model.train()
        pending = checkpoint["training_log_state"] if checkpoint else {}
        accumulated, count = pending.get("loss_sum", 0.0), pending.get("batches", 0)
        grad_sum, grad_max = pending.get("grad_sum", 0.0), pending.get("grad_max", 0.0)
        component_sums = pending.get("loss_components", {})
        completed_steps, stop_reason = start, "max_steps"
        max_steps = cfg.training.max_steps
        session_start = window_start = time.perf_counter()
        window_steps, data_seconds, validation_seconds = 0, 0.0, 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        log.info("Training from step %d to at most %d…", start, max_steps)
        fetch_start = time.perf_counter()
        for step, x in enumerate(loader, start):
            data_seconds += time.perf_counter() - fetch_start
            x = x.to(device, non_blocking=True)
            lr = learning_rate(step, cfg.training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if objective is None:
                reconstructed, _ = model(x)
                loss = mse = F.mse_loss(reconstructed, x)
            else:
                loss, components = objective(model, x)
                mse = components["mse"]
                for name, value in components.items():
                    if name != "mse":
                        component_sums[name] = component_sums.get(name, 0.0) + value.detach().item()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip,
                                                  error_if_nonfinite=True)
            optimizer.step()
            grad_value = grad.item()
            accumulated += mse.item()
            count += 1
            grad_sum += grad_value
            grad_max = max(grad_max, grad_value)
            window_steps += 1
            done = step + 1
            completed_steps = done
            validation_record, should_stop = None, False
            if done % cfg.validation.every_steps == 0 or done == max_steps:
                validation_start = time.perf_counter()
                validation_record = (validate(model, monitor, cfg.validation.batch_size, device)
                                     if objective is None else
                                     validate(model, monitor, cfg.validation.batch_size, device, objective, cfg.zstd))
                validation_seconds = time.perf_counter() - validation_start
                validation_duration = validation_seconds
                should_stop = early_stopping.update(validation_record["val_psnr"])
            # Flush the final training window even when stopping between logging intervals.
            if done % cfg.tracking.log_every_steps == 0 or done == max_steps or should_stop:
                mean_mse = accumulated / count
                record = {"train_mse": mean_mse, "train_psnr": -10 * math.log10(mean_mse),
                          "learning_rate": lr, "gradient_norm": grad_value,
                          "gradient_norm_mean": grad_sum / count, "gradient_norm_max": grad_max}
                record.update({f"train_{name}": value / count for name, value in component_sums.items()})
                mlflow.log_metrics(record, step=done)
                history.append({"step": done, **record})
                now = time.perf_counter()
                elapsed = now - window_start - validation_seconds
                speed = window_steps / max(elapsed, 1e-9)
                timing = {"step": done, "steps_per_second": speed,
                          "images_per_second": speed * cfg.training.batch_size,
                          "data_wait_fraction": min(1.0, data_seconds / max(elapsed, 1e-9)),
                          "session_elapsed_seconds": now - session_start}
                if device.type == "cuda":
                    timing["gpu_memory_allocated_gib"] = torch.cuda.memory_allocated(device) / 2 ** 30
                    timing["gpu_memory_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                    torch.cuda.reset_peak_memory_stats(device)
                performance.append(timing)
                mlflow.log_metrics({f"perf_{k}": v for k, v in timing.items() if k != "step"}, step=done)
                gpu = (f" | GPU mem {timing['gpu_memory_allocated_gib']:.2f}/{timing['gpu_memory_peak_gib']:.2f} GiB"
                       if device.type == "cuda" else "")
                log.info("[train] step %d/%d (%5.1f%%) | mse %.6f (%.2f dB) | lr %.2e | grad mean %.3f max %.3f"
                         " | %.2f it/s, %.1f crops/s | data wait %2.0f%%%s | elapsed %s ETA %s",
                         done, max_steps, 100 * done / max_steps, mean_mse, record["train_psnr"], lr,
                         record["gradient_norm_mean"], grad_max, speed, timing["images_per_second"],
                         100 * timing["data_wait_fraction"], gpu, clock(now - session_start),
                         clock((max_steps - done) / max(speed, 1e-9)))
                accumulated, count, grad_sum, grad_max = 0.0, 0, 0.0, 0.0
                component_sums = {}
                window_start, window_steps, data_seconds, validation_seconds = now, 0, 0.0, 0.0
            if validation_record is not None:
                record = {**validation_record,
                          "early_stopping_bad_checks": early_stopping.bad_checks,
                          "early_stopping_reference_psnr": early_stopping.reference_psnr}
                mlflow.log_metrics(record, step=done)
                previous = [r["val_psnr"] for r in history if "val_psnr" in r]
                history.append({"step": done, **record})
                improved = record["val_psnr"] > best
                best = max(best, record["val_psnr"])
                state = {"loss_sum": accumulated, "batches": count, "grad_sum": grad_sum, "grad_max": grad_max}
                if objective is not None:
                    state["loss_components"] = component_sums.copy()
                if improved:
                    save_checkpoint(out / "best.pt", model, optimizer, done, best, cfg, data_manifest, history,
                                    early_stopping, state, performance, calibration)
                    mlflow.log_artifact(str(out / "best.pt"), "checkpoints")
                save_checkpoint(out / "last.pt", model, optimizer, done, best, cfg, data_manifest, history,
                                early_stopping, state, performance, calibration)
                mlflow.log_artifact(str(out / "last.pt"), "checkpoints")
                delta = f"{record['val_psnr'] - previous[-1]:+.3f}" if previous else "first"
                extra = (f" (min {record['val_psnr_min']:.2f}, max {record['val_psnr_max']:.2f})"
                         if "val_psnr_min" in record else "")
                latent = (f" | latent std {record['latent_std']:.3f}, |z| p99.9 {record['latent_abs_p999']:.3f}"
                          if "latent_std" in record else "")
                log.info("[val]   step %d/%d | PSNR %.3f dB (Δ %s)%s | SSIM %.5f | best %.3f dB%s | patience "
                         "%d/%d (ref %.3f dB)%s | %.1fs", done, max_steps, record["val_psnr"], delta, extra,
                         record["val_ssim"], best, " ★ new best, saved best.pt" if improved else "",
                         early_stopping.bad_checks, early_stopping.patience, early_stopping.reference_psnr,
                         latent, validation_duration)
                if objective is not None:
                    log.info("[rate]  step %d | measured int8 Zstd %.4f bpp | MSE + λ × Zstd bpp %.6f | clip %.3f%%",
                             done, record["val_zstd_bpp"], record["val_rate_distortion"],
                             100 * record["val_quantization_clipped_fraction"])
                examples = (reconstruction_examples(model, sample_images, sample_names, device, objective)
                            if cfg.tracking.plots and sample_count else [])
                refresh_plots(cfg, out, history, performance, examples, done)
                if should_stop and done < max_steps:
                    stop_reason = "early_stopping"
                    log.info("Early stopping at step %d: %d validation checks without a PSNR gain > %s dB.",
                             done, early_stopping.bad_checks, early_stopping.min_delta)
                    break
            fetch_start = time.perf_counter()
        # Keep the best artifact accessible locally after a resume even if it never improved.
        if not (out / "best.pt").exists():
            mlflow.artifacts.download_artifacts(run_id=mlflow.active_run().info.run_id,
                                               artifact_path="checkpoints/best.pt", dst_path=str(out))
        write_json(out / "history.json", history)
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in history for k in row}))
            writer.writeheader()
            writer.writerows(history)
        write_json(out / "performance.json", performance)
        for name in ["history.json", "history.csv", "performance.json"]:
            mlflow.log_artifact(str(out / name))
        mlflow.set_tags({"stop_reason": stop_reason, "stopped_early": str(stop_reason == "early_stopping").lower()})
        mlflow.log_metrics({"completed_steps": completed_steps, "best_val_psnr": best,
                            "stopped_early": int(stop_reason == "early_stopping")}, step=completed_steps)
        best_step = max((r for r in history if "val_psnr" in r), key=lambda r: r["val_psnr"], default={})
        write_json(out / "training_summary.json", {"steps": completed_steps,
                   "max_steps": cfg.training.max_steps, "stop_reason": stop_reason,
                   "stopped_early": stop_reason == "early_stopping", "early_stopping": early_stopping.state_dict(),
                   "early_stopping_config": OmegaConf.to_container(cfg.training.early_stopping),
                   "gradient_clip": cfg.training.gradient_clip,
                   "best_val_psnr": best, "best_step": best_step.get("step"),
                   "best_val_ssim": best_step.get("val_ssim"),
                   "loss": cfg.training.loss,
                   "best_val_zstd_bpp": best_step.get("val_zstd_bpp"),
                   "best_val_rate_distortion": best_step.get("val_rate_distortion"),
                   "session_seconds": time.perf_counter() - session_start,
                   "run_id": mlflow.active_run().info.run_id})
        mlflow.log_artifact(str(out / "training_summary.json"))
        refresh_plots(cfg, out, history, performance, [], completed_steps, final=True)
        rule = "=" * 88
        log.info(rule)
        log.info("Finished: %d/%d steps, stop reason '%s', session time %s", completed_steps, max_steps,
                 stop_reason, clock(time.perf_counter() - session_start))
        if best_step:
            log.info("Best validation: %.3f dB PSNR, SSIM %.5f at step %d -> %s", best,
                     best_step["val_ssim"], best_step["step"], out / "best.pt")
        log.info("Plots: %s", out / "plots")
        log.info("Next: python evaluate.py evaluation.checkpoint=%s", (out / "best.pt").resolve())
        log.info(rule)
        log_file = out / f"{HydraConfig.get().job.name}.log"
        if log_file.is_file():
            mlflow.log_artifact(str(log_file), "logs")
        print(json.dumps({"output_dir": str(out), "run_id": mlflow.active_run().info.run_id}), flush=True)


if __name__ == "__main__":
    main()
