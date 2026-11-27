# Zstd-Optimized Autoencoder

Hydra defines experiments; MLflow records their configurations, source files, environment,
training curves, checkpoints, and evaluation artifacts.

## Setup

Use Python 3.12 and install an appropriate PyTorch build for the target machine, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset layout relative to `data.root` (default: this repository):

```text
DIV2K_train_HR/                # 548 PNGs
Datasets/DIV2K_train_HR/       # remaining 252 training PNGs
Datasets/DIV2K_valid_HR/       # 100 final test PNGs
```

Override `data.root=/path/to/data` or set `DIV2K_ROOT` on another PC. Paths and runtime
settings may differ between machines; the saved image lists and content hashes identify
the exact dataset and split. Twenty of the 800 training images form an internal validation
set. The 100 DIV2K valid images are reserved for final testing.

## Verify, train, evaluate

Inspect configuration without training:

```bash
python train.py --cfg job --resolve
```

Verify the pipeline first (two updates with a deliberately small model):

```bash
python -m unittest discover -s tests -v
python train.py --config-name smoke
python evaluate.py --config-name smoke evaluation.checkpoint=/absolute/path/from/training/best.pt
```

Smoke evaluation uses two validation crops and calibration uses two training crops. It
does not evaluate final test images. Smoke runs are tagged `scientific_result=false` and
must not be interpreted as baseline quality or compression results.

Run the agreed full baseline when ready (this starts the long training job):

```bash
python train.py
python evaluate.py evaluation.checkpoint=/absolute/path/from/training/best.pt
```

Training uses the 128-hidden/64-latent-channel residual model with four downsampling stages,
256-square random crops, MSE, Adam, batch size 16, and 40,000 updates. Warmup lasts 1,000
updates; cosine decay ends at a learning rate of 1e-6. Every 2,000 updates, fixed 512-square
validation crops select the best checkpoint by mean PSNR. Numerical precision is float32.
Training samples images with replacement and draws a new crop per sample. The crop stream
is index-seeded, so resuming an interrupted job continues the same samples.

Resume with the same scientific configuration and the latest checkpoint:

```bash
python train.py training.resume_checkpoint=/absolute/path/to/last.pt
```

This continues the same MLflow run; it does not reset the optimizer or scheduler. Changing
architecture, training settings, seed, image contents, or splits is rejected. Resume supports
interruption of the configured budget, not changing the budget of an already completed run.
Hardware changes can introduce floating-point differences.

## Outputs and MLflow

Each invocation prints its output directory and run ID. Hydra writes under
`outputs/<experiment>/<timestamp>/`, and training saves `best.pt`, `last.pt`, and history
files there. Evaluation creates a separate run tagged with the training run ID and exact
checkpoint hash. Use a validation evaluation during development:

```bash
python evaluate.py evaluation.checkpoint=/path/to/best.pt evaluation.split=validation
```

MLflow uses `sqlite:///mlflow.db` and `mlartifacts/` locally. From the repository root:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The Python client can log locally without running this server.
Set `MLFLOW_TRACKING_URI` to use an already configured shared server. Local artifact paths
belong to the machine that created them; for transfer, preserve the database and artifacts
and account for their recorded paths, or use a shared server from the beginning.

## Measurement protocol

`per_image.csv` contains one row per image and latent representation (`float32` or `int8`).
Each row measures storage both without and with Zstd, giving the four planned combinations:

- Original PNG bytes and uncompressed RGB8 bytes.
- Numeric latent payload bytes, complete uncompressed latent-file bytes, and complete
  compressed latent-file bytes (including metadata).
- Bits per original image pixel before/after Zstd and compression ratios relative to
  both original PNG and raw RGB bytes. A ratio above 1 means the latent file is smaller.
- Reconstructed PNG bytes with fixed PNG settings, PSNR, and SSIM against original RGB8.
- Quantization clipping fraction and an assertion of exact Zstd byte restoration.

Quantization defaults are **provisional**: symmetric per-channel int8, range -127 to 127,
with scales calibrated at the 99.9th percentile of absolute latent values on fixed crops
from the first 20 **training** images. Test images never choose scales. `quantization.json`
records those scales and their calibration provenance. To freeze the same quantizer for
future A/B evaluations, reuse the baseline's calibration file:

```bash
python evaluate.py evaluation.checkpoint=/path/to/best.pt quantization.calibration_file=/path/to/quantization.json
```

Zstd defaults are level 3, no dictionary, single-threaded, checksums enabled. Latents are
CHW in C order; floats are little-endian float32. The versioned file includes latent shape,
original dimensions, checkpoint hash, and quantization scales when applicable. Decoding
requires the corresponding shared decoder weights, whose size is reported separately.
Compressed and uncompressed files restore identical latent bytes, so each representation
has the same reconstruction quality with and without Zstd. Fewer float32 values alone do
not guarantee a smaller file than PNG.

Quality is evaluated after clipping to [0,1] and rounding to RGB8. SSIM uses an 11x11 Gaussian
window, sigma 1.5, valid region, RGB channel mean, and data range 1. Images are replicate-padded
to a multiple of 16 before encoding and cropped to original dimensions after decoding.
Latent storage includes padding; bpp uses the original image area. Summary PSNR/SSIM are
image means; summary bpp and compression ratios use total bytes and total original pixels.
Reconstructed PNG size is an output diagnostic, not the compressed bitstream size.

The pretrained-codec benchmark and Zstd-aware loss are later phases; they are not yet
implemented. Start with a verified baseline before making those scientific comparisons.
