# Zstd-Optimized Autoencoder

Hydra defines experiments; MLflow records their configurations, source files, environment,
training curves, checkpoints, and evaluation artifacts.

## Current scope

The best recorded baseline is [v3](results/baseline_v3/README.md): **33.05 dB PSNR,
0.916 SSIM, 1.531 bpp** for int8 + Zstd on 100 full-resolution test images.
Its trained weights are not included in Git.

| Configuration | Training | Evaluation |
|---|---|---|
| `configs/config.yaml` | One baseline, up to 5,000 updates with early stopping | One saved checkpoint on 100 full-resolution test images |
| `configs/smoke.yaml` | Two updates with a smaller model; early stopping disabled | Two 32×32 validation crops; verification only |
| `configs/baseline_v3.yaml` | Best baseline, 30,000-step cap | Four storage conditions on 100 full-resolution test images |
| `configs/zstd.yaml` | V3 with only the training loss changed | The same v3 evaluation pipeline |

The first baseline requires **one training run and one evaluation run** in MLflow.
Training and in-training validation use continuous float32 latents. **Quantization and
Zstd are applied only by `evaluate.py`**; `quantization.enabled` controls that evaluation
branch. The same checkpoint supplies all four storage conditions: float32/int8, each
without/with Zstd. Quantization-aware reconstruction and pretrained-codec benchmarks
remain future work. The Zstd-size experiment below changes only the training loss.

## Actual Zstd-size loss

`configs/zstd.yaml` inherits v3's architecture, model forward pass, crop stream,
optimizer, schedule, bf16/compilation settings, validation, and checkpoint selection.
The decoder receives the original **continuous latent**. The training objective is:

```text
loss = reconstruction_MSE + lambda_rate * actual_Zstd_bpp
actual_Zstd_bpp = mean_i(8 * Zstd_latent_file_bytes_i / input_pixels_i)
```

For the size term only, each image's latent is copied, quantized to int8 with the
reference scales in `results/baseline_v3/evaluation/quantization.json`, and passed
through the existing `zae.codec.serialize` with Zstd. Complete file bytes include
the existing metadata. Larger files directly increase the loss; there are no entropy,
smoothness, or overflow proxy terms and no quantization-aware reconstruction.
The reference scales are a loss parameter under `training.zstd_loss`; they do not
change v3's evaluation calibration settings.

Zstd is discrete, so backward uses **two-point Gaussian finite differences** of
actual compressed sizes. With `u = z / scales`, perturbations `e ~ N(0,I)`, and
`f(u)` the measured file bpp, the estimator averages:

```text
gradient_z ≈ (f(u + sigma*e) - f(u - sigma*e)) * e / (2*sigma*scales)
```

The forward value is the measured unperturbed size. The estimated gradient targets
its Gaussian-smoothed size; it is not an exact derivative of discrete Zstd bytes.
This is a [Gaussian random-search estimator](https://proceedings.mlr.press/v162/gao22f.html).
Defaults: `lambda_rate=0.001`, `perturbation_symbols=1.0`, `directions=4`. Each image
requires one compression for the value and eight probe compressions for backward.
Sampling uses the torch RNG already saved by v3, preserving interrupted-run resume.
The estimator has sampling variance and smoothing bias; compression gains require
actual rate–distortion evaluation. Loss and measured bpp are logged separately from MSE.

```bash
python train.py --config-name zstd runtime.device=cuda
python evaluate.py --config-name zstd evaluation.checkpoint=/path/to/best.pt
```

For a matched comparison using the same quantizer as the size term, use v3's
existing evaluation option for both the baseline and the new model:

```bash
python evaluate.py --config-name zstd evaluation.checkpoint=/path/to/best.pt quantization.calibration_file=results/baseline_v3/evaluation/quantization.json
```

`training.zstd_loss.lambda_rate=0` retains v3's reconstruction MSE and gradients.
Training starts from v3's seeded initialization; this configuration does not load
the absent trained v3 checkpoint.

See the [file map and pipeline diagram](documentation/file-map.md) for each file's role.

## Setup

Use Python 3.12 and install an appropriate PyTorch build for the target machine, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset layout relative to `data.root` (default: this repository):

```text
Datasets/DIV2K_train_HR/       # 800 PNGs: 780 training + 20 internal validation
Datasets/DIV2K_valid_HR/       # 100 final test PNGs
```

Override `data.root=/path/to/data` or set `DIV2K_ROOT` on another PC. Paths and runtime
settings may differ between machines; the saved image lists and content hashes identify
the exact dataset and split. Twenty of the 800 training images form an internal validation
set. The 100 DIV2K valid images are reserved for final testing.
The [dataset documentation](documentation/file-map.md#dataset) contains official download links.

## Verify, train, evaluate

Inspect configuration without training:

```bash
python train.py --cfg job --resolve
```

Optional verification (the test suite runs tiny synthetic training jobs; the smoke
training command performs two updates on real training crops):

```bash
python -m unittest discover -s tests -v
python train.py --config-name smoke
python evaluate.py --config-name smoke evaluation.checkpoint=/absolute/path/from/training/best.pt
```

Run commands from the repository root. Replace example checkpoint paths with
`<output_dir>/best.pt`, using the output directory printed by training. Use the same model and dataset settings
for evaluation as for training.

The smoke config disables early stopping to keep its verification run at two updates.
Its in-training validation uses all 20 validation images as fixed 32×32 crops;
its separate evaluation uses only two validation crops and calibration uses two training crops. It
does not evaluate final test images. Smoke runs are tagged `scientific_result=false` and
must not be interpreted as baseline quality or compression results.

Run the agreed full baseline when ready (this starts the long training job):

```bash
python train.py
python evaluate.py evaluation.checkpoint=/absolute/path/from/training/best.pt
```

Training uses the 128-hidden/64-latent-channel residual model with four downsampling stages,
256-square random crops, MSE, Adam, batch size 16, and at most 5,000 updates. Warmup lasts 250
updates; cosine decay is scheduled to reach 1e-6 at the step cap. Every 500 updates, fixed 512-square
validation crops select the best checkpoint by mean PSNR. Numerical precision is float32.
Training samples images with replacement and draws a new crop per sample. The crop stream
is index-seeded, so resuming an interrupted job continues the same samples.
The first-run cap processes 80,000 crops, approximately 103 crop visits per training image.
This is a starting budget; validation curves will show whether more training is justified.

### Early stopping

`training.early_stopping` enables validation-based stopping by default:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.01
```

The first validation PSNR establishes a reference. A gain **greater than 0.01 dB**
over the last significant reference resets patience; five consecutive validation checks
without such a gain stop training. At the default validation interval, this is 2,500
updates after the last significant improvement. Small gains can accumulate against the
reference. `max_steps` remains the upper limit; the learning-rate schedule is not shortened
when early stopping occurs. To disable the rule, override `training.early_stopping.enabled=false`.

`best.pt` always stores the highest validation PSNR, even when its improvement is smaller
than `min_delta`. `last.pt` stores the latest training state and patience counter.
`training_summary.json` and MLflow record the actual completed steps, configured cap,
stop reason (`early_stopping` or `max_steps`), and best validation PSNR. Training loss is
flushed at the stopping step even if it falls between logging intervals.

Resume with the same scientific configuration and the latest checkpoint:

```bash
python train.py training.resume_checkpoint=/absolute/path/to/last.pt
```

This continues the same MLflow run; it does not reset the optimizer or scheduler. Changing
architecture, training settings, seed, image contents, or splits is rejected. Resume supports
interruption of the configured budget, not changing the budget of an already completed run.
The patience counter and significant PSNR reference are restored, along with any pending
training-loss logging window. Checkpoints already marked as early-stopped cannot resume;
changing the stopping policy requires a new experiment.
Hardware changes can introduce floating-point differences.

## Outputs and MLflow

Successful training and evaluation invocations print their output directory and run ID. Hydra writes under
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
With the default settings this produces **200 rows: 100 images × 2 representations**.
Each row measures storage both without and with Zstd, giving four storage conditions
inside a single evaluation run:

- Original PNG bytes and uncompressed RGB8 bytes.
- Numeric latent payload bytes, complete uncompressed latent-file bytes, and complete
  compressed latent-file bytes (including metadata).
- Bits per original image pixel before/after Zstd and compression ratios relative to
  both original PNG and raw RGB bytes. A ratio above 1 means the latent file is smaller.
- Reconstructed PNG bytes with fixed PNG settings, PSNR, and SSIM against original RGB8.
- Quantization clipping fraction and an assertion of exact Zstd byte restoration.

For cropped smoke evaluation, dimensions and measurements refer to the crop. Its
`original_png_bytes` measures a PNG of that crop; `source_png_bytes` retains the full source
file's size. These crop results are marked with `is_crop=true`.

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
