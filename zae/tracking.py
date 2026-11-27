import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf

from zae.early_stopping import EarlyStopping


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, key))
        else:
            result[key] = json.dumps(item) if isinstance(item, list) else str(item)
    return result


def setup_tracking(cfg):
    uri = cfg.tracking.uri
    if uri.startswith("sqlite:///"):
        uri = "sqlite:///" + str(Path(uri.removeprefix("sqlite:///")).expanduser().resolve())
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(cfg.tracking.experiment_name)
    if experiment is None:
        location = cfg.tracking.artifact_location
        if location is None and uri.startswith("sqlite:"):
            location = Path("mlartifacts").resolve().as_uri()
        elif location is not None and "://" not in location:
            location = Path(location).expanduser().resolve().as_uri()
        # A remote server must choose its own artifact location unless explicitly overridden.
        experiment_id = mlflow.create_experiment(cfg.tracking.experiment_name, artifact_location=location)
        experiment = mlflow.get_experiment(experiment_id)
    mlflow.set_experiment(experiment_id=experiment.experiment_id)


def log_context(cfg, out, data_manifest, kind, device):
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), out / "config.yaml")
    write_json(out / "split_manifest.json", data_manifest)
    packages = {}
    for package in ["torch", "numpy", "Pillow", "hydra-core", "omegaconf", "mlflow", "zstandard"]:
        packages[package] = importlib.metadata.version(package)

    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    # Include untracked source files: the initial implementation may not be committed yet.
    sources = [Path("train.py"), Path("evaluate.py"), Path("requirements.txt")]
    sources += sorted(Path("zae").glob("*.py")) + sorted(Path("configs").glob("*.yaml"))
    source_hashes = {p.as_posix(): sha256(p) for p in sources if p.is_file()}
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": packages, "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "source_sha256": source_hashes,
    }
    write_json(out / "environment.json", environment)
    write_json(out / "run.json", {"run_id": mlflow.active_run().info.run_id,
                                  "tracking_uri": mlflow.get_tracking_uri(), "output_dir": str(out)})
    mlflow.set_tags({"run_kind": kind, "dataset_id": data_manifest["dataset_id"],
                     "description": cfg.experiment.description,
                     "scientific_result": str(cfg.experiment.name != "smoke").lower()})
    # Resume starts the same MLflow run; previously recorded configuration stays immutable.
    params = flatten(resolved)
    params.pop("training.resume_checkpoint", None)
    existing = mlflow.get_run(mlflow.active_run().info.run_id).data.params
    mlflow.log_params({k: v for k, v in params.items() if k not in existing})
    for name in ["config.yaml", "split_manifest.json", "environment.json", "run.json"]:
        mlflow.log_artifact(str(out / name))
    for p in sources:
        if p.is_file():
            mlflow.log_artifact(str(p), artifact_path=(Path("source") / p.parent).as_posix())


def validate_config(cfg):
    EarlyStopping(**OmegaConf.to_container(cfg.training.early_stopping))
    expected = {
        "training.optimizer": "adam", "training.scheduler": "cosine", "training.loss": "mse",
        "validation.checkpoint_metric": "psnr", "validation.checkpoint_mode": "max",
        "runtime.precision": "float32", "quantization.scheme": "uniform_symmetric_per_channel",
        "quantization.dtype": "int8", "serialization.format_version": 1,
        "serialization.float_dtype": "float32", "serialization.layout": "CHW",
        "serialization.byte_order": "little", "zstd.dictionary": None,
        "evaluation.padding_mode": "replicate", "evaluation.image_rounding": "uint8",
    }
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"{key}: only {value!r} is implemented")
    for key in ["training.batch_size", "training.max_steps", "validation.every_steps",
                "validation.batch_size", "tracking.log_every_steps", "runtime.cpu_threads"]:
        if OmegaConf.select(cfg, key) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= cfg.training.warmup_steps < cfg.training.max_steps:
        raise ValueError("warmup_steps must be >= 0 and < max_steps")
    if not 0 < cfg.training.minimum_learning_rate <= cfg.training.learning_rate:
        raise ValueError("Invalid learning rate range")
    if cfg.data.crop_size % (2 ** cfg.model.downsampling_stages):
        raise ValueError("Training crop must be divisible by the downsampling factor")
    if not 1 <= cfg.quantization.qmax <= 127:
        raise ValueError("int8 qmax must be between 1 and 127")
    if not 0 < cfg.quantization.calibration_percentile <= 100:
        raise ValueError("Invalid calibration percentile")
    if cfg.quantization.calibration_images < 1 or cfg.runtime.num_workers < 0:
        raise ValueError("Invalid calibration count/worker count")
    if cfg.evaluation.split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if cfg.evaluation.max_images is not None and cfg.evaluation.max_images < 1:
        raise ValueError("max_images must be positive or null")
    if not 0 <= cfg.evaluation.png_compress_level <= 9:
        raise ValueError("PNG compression level must be between 0 and 9")
