"""Train a baseline; Hydra config and all run artifacts are recorded in MLflow."""
import csv
import json
import math
import random
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
from zae.metrics import encode_padded, psnr, round_rgb, ssim
from zae.model import AutoEncoder
from zae.tracking import (choose_device, log_context, seed_everything,
                          setup_tracking, validate_config, write_json)


def learning_rate(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    # Reaches the configured minimum on the last update.
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps - 1)
    return cfg.minimum_learning_rate + 0.5 * (cfg.learning_rate - cfg.minimum_learning_rate) * (
        1 + math.cos(math.pi * min(progress, 1)))


@torch.inference_mode()
def validate(model, images, batch_size, device):
    model.eval()
    values = []
    for start in range(0, len(images), batch_size):
        x = torch.stack(images[start:start + batch_size]).to(device)
        h, w = x.shape[-2:]
        reconstruction = round_rgb(model.decode(encode_padded(model, x))[..., :h, :w])
        values.extend(zip(psnr(reconstruction, x).tolist(), ssim(reconstruction, x).tolist()))
    model.train()
    return {"val_psnr": sum(p for p, _ in values) / len(values),
            "val_ssim": sum(s for _, s in values) / len(values)}


def save_checkpoint(path, model, optimizer, step, best, cfg, data_manifest, history):
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "best_psnr": best, "model_config": OmegaConf.to_container(cfg.model),
        "config": OmegaConf.to_container(cfg, resolve=True), "manifest": data_manifest,
        "history": history, "run_id": mlflow.active_run().info.run_id,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
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


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    validate_config(cfg)
    seed_everything(cfg.experiment.seed, cfg.runtime.cpu_threads)
    device = choose_device(cfg.runtime.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    data_manifest = manifest(cfg)
    model = AutoEncoder(**OmegaConf.to_container(cfg.model)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    start, best, history, checkpoint = 0, -float("inf"), [], None
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
        if start >= cfg.training.max_steps:
            raise ValueError("Checkpoint already reached max_steps; no training is needed")
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    monitor = [center_crop(load_rgb(p), cfg.validation.crop_size)
               for p in split_paths(cfg, data_manifest, "validation")]
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
    with mlflow.start_run(**run_args):
        log_context(cfg, out, data_manifest, "training", device)
        mlflow.log_metric("model_parameters", sum(p.numel() for p in model.parameters()))
        mlflow.log_metric("model_weight_bytes", sum(p.numel() * p.element_size() for p in model.parameters()))
        mlflow.log_metric("decoder_weight_bytes", sum(p.numel() * p.element_size() for p in model.decoder.parameters()))
        model.train()
        accumulated, count = 0.0, 0
        for step, x in enumerate(loader, start):
            x = x.to(device)
            lr = learning_rate(step, cfg.training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(x)
            loss = F.mse_loss(reconstructed, x)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip,
                                                  error_if_nonfinite=True)
            optimizer.step()
            accumulated += loss.item()
            count += 1
            done = step + 1
            if done % cfg.tracking.log_every_steps == 0 or done == cfg.training.max_steps:
                record = {"train_mse": accumulated / count, "learning_rate": lr, "gradient_norm": grad.item()}
                mlflow.log_metrics(record, step=done)
                history.append({"step": done, **record})
                accumulated, count = 0.0, 0
            if done % cfg.validation.every_steps == 0 or done == cfg.training.max_steps:
                record = validate(model, monitor, cfg.validation.batch_size, device)
                mlflow.log_metrics(record, step=done)
                history.append({"step": done, **record})
                improved = record["val_psnr"] > best
                best = max(best, record["val_psnr"])
                if improved:
                    save_checkpoint(out / "best.pt", model, optimizer, done, best, cfg, data_manifest, history)
                    mlflow.log_artifact(str(out / "best.pt"), "checkpoints")
                save_checkpoint(out / "last.pt", model, optimizer, done, best, cfg, data_manifest, history)
                mlflow.log_artifact(str(out / "last.pt"), "checkpoints")
                print(f"step={done} val_psnr={record['val_psnr']:.3f} val_ssim={record['val_ssim']:.5f}", flush=True)
        # Keep the best artifact accessible locally after a resume even if it never improved.
        if not (out / "best.pt").exists():
            mlflow.artifacts.download_artifacts(run_id=mlflow.active_run().info.run_id,
                                               artifact_path="checkpoints/best.pt", dst_path=str(out))
        write_json(out / "history.json", history)
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in history for k in row}))
            writer.writeheader()
            writer.writerows(history)
        mlflow.log_artifact(str(out / "history.json"))
        mlflow.log_artifact(str(out / "history.csv"))
        write_json(out / "training_summary.json", {"steps": cfg.training.max_steps,
                   "best_val_psnr": best, "run_id": mlflow.active_run().info.run_id})
        mlflow.log_artifact(str(out / "training_summary.json"))
        print(json.dumps({"output_dir": str(out), "run_id": mlflow.active_run().info.run_id}), flush=True)


if __name__ == "__main__":
    main()
