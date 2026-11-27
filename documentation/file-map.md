# Baseline experiment file map

All paths below are relative to the project root. The baseline is still untrained;
the existing smoke outputs come from small pipeline verification runs.

## Files created

| File | What it does |
|---|---|
| [`configs/config.yaml`](../configs/config.yaml) | Main Hydra configuration: dataset split, architecture, training, validation, quantization, Zstd, evaluation, and MLflow settings. Contains the proposed 40,000-step baseline plan. |
| [`configs/smoke.yaml`](../configs/smoke.yaml) | Overrides the main config for verification: smaller model, two updates, and two validation crops for evaluation. Running it performs a small training check. |
| [`configs/__init__.py`](../configs/__init__.py) | Makes the configuration folder importable so Hydra also works when entry points are imported. |
| [`train.py`](../train.py) | Training entry point. Loads Hydra settings, trains the autoencoder, validates it, saves best/latest checkpoints, supports resume, and logs results to MLflow. |
| [`evaluate.py`](../evaluate.py) | Evaluation entry point. Loads a checkpoint, evaluates float32 and quantized latents before/after Zstd, and saves per-image measurements, summaries, latent files, and reconstructed PNGs. Calibrates quantization using training images. |
| [`zae/model.py`](../zae/model.py) | Defines the agreed residual convolutional autoencoder and PixelShuffle decoder. |
| [`zae/data.py`](../zae/data.py) | Loads RGB images, creates the fixed train/validation/test split, hashes dataset contents, and generates reproducible training crops. |
| [`zae/metrics.py`](../zae/metrics.py) | Calculates PSNR and SSIM; rounds reconstructions to RGB8 and pads inputs for the encoder. |
| [`zae/codec.py`](../zae/codec.py) | Quantizes latents, writes/reads versioned latent files with metadata, and compresses/decompresses their payloads using Zstd. |
| [`zae/tracking.py`](../zae/tracking.py) | Connects to MLflow, records resolved configs and source/environment information, sets seeds/devices, and checks supported configuration values. |
| [`zae/__init__.py`](../zae/__init__.py) | Marks the shared implementation as a Python package. |
| [`tests/test_pipeline.py`](../tests/test_pipeline.py) | Checks model shapes, metrics, quantization, exact latent-file round trips, configuration values, resume safeguards, and summary calculations. |
| [`tests/test_training_resume.py`](../tests/test_training_resume.py) | Runs tiny synthetic training jobs to verify that interrupted/resumed CPU training matches uninterrupted training exactly. |
| [`requirements.txt`](../requirements.txt) | Lists the Python dependencies and supported version ranges. |
| [`README.md`](../README.md) | Explains setup, commands, outputs, and the measurement protocol. |
| [`.gitignore`](../.gitignore) | Keeps datasets, environments, caches, generated results, and local MLflow storage out of Git. |
| [`documentation/file-map.md`](file-map.md) | This map. |

## Generated files and folders

| Location | Contents |
|---|---|
| `outputs/<experiment>/<timestamp>/` | One invocation's output folder. Contains `config.yaml`, `split_manifest.json`, `environment.json`, `run.json`, and Hydra's `.hydra/` configuration snapshots. |
| Training output folder | `best.pt`, `last.pt`, `history.csv`, `history.json`, and `training_summary.json`. |
| Evaluation output folder | `per_image.csv`, `summary.json`, and `quantization.json` when quantization is enabled. `latents/` holds plain/compressed latent files; `reconstructions/` holds output PNGs. |
| `mlflow.db` | Local MLflow database containing run metadata, parameters, metrics, and artifact references. |
| `mlartifacts/` | MLflow's saved copies of source files, configurations, checkpoints, and result files. |

The per-image CSV records original PNG/raw RGB sizes, latent payload/file sizes,
Zstd file sizes, reconstruction PNG sizes, PSNR, SSIM, bpp, compression ratios,
quantization clipping, and exact Zstd restoration checks.

The project specification PDF and datasets remain in their original locations.
