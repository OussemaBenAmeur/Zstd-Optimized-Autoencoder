# Zstd-Optimized Autoencoder

Hydra defines experiments; MLflow records their configurations, source files, environment,
training curves, checkpoints, and evaluation artifacts.

## Current scope

The baseline is untrained; completed runs are pipeline verification only.

| Configuration | Training | Evaluation |
|---|---|---|
| `configs/config.yaml` | One baseline, up to 5,000 updates, then review validation quality | After review, one saved checkpoint on 100 full-resolution test images |
| `configs/smoke.yaml` | Two updates with a smaller model; early stopping disabled | Two 32×32 validation crops; verification only |

An accepted baseline requires **one training run and one evaluation run** in MLflow.
First review the training run's validation PSNR, SSIM, and saved reconstruction examples.
Final evaluation is a separate, manually launched step after that quality review.
Training and in-training validation use continuous float32 latents. **Quantization and
Zstd are applied only by `evaluate.py`**; `quantization.enabled` controls that evaluation
branch. The same checkpoint supplies all four storage conditions: float32/int8, each
without/with Zstd. Quantization-aware training, Zstd-aware training, and pretrained-codec
benchmarks are future work.

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

Train the baseline first (this starts training, without launching final evaluation):

```bash
python train.py
```

Training uses the 128-hidden/64-latent-channel residual model with four downsampling stages,
256-square random crops, MSE, Adam, batch size 16, and at most 5,000 updates. Warmup lasts 250
updates; cosine decay is scheduled to reach 1e-6 at the step cap. Every 500 updates, fixed 512-square
validation crops select the best checkpoint by mean PSNR. Numerical precision is float32.
Training samples images with replacement and draws a new crop per sample. The crop stream
is index-seeded, so resuming an interrupted job continues the same samples.
The first-run cap processes 80,000 crops, approximately 103 crop visits per training image.
This is a starting budget; validation curves will show whether more training is justified.

### Review validation reconstruction before evaluation

Whenever validation PSNR improves, training saves up to four fixed validation image pairs
in `validation_examples/` and logs them to the training run in MLflow. **Original is on the
left; reconstruction is on the right.** These use continuous latents and 512×512 crops.
`index.json` records the checkpoint step, image indices in `split_manifest.json`, and pair metrics.
The previews correspond to the best checkpoint, not necessarily the last training step.

Review these images together with validation PSNR/SSIM and the learning curves. Early
stopping or reaching 5,000 steps does not establish acceptable reconstruction quality.
If quality is insufficient, improve the model or training before testing. There is no
automatic numeric acceptance threshold yet; this is an explicit review decision.

Only after accepting validation reconstruction, run compression evaluation:

```bash
python evaluate.py evaluation.checkpoint=/absolute/path/from/training/best.pt
```

Training never launches this command automatically. The 100 test images stay reserved
until the model is ready. Use `evaluation.split=validation` when tuning compression
settings before final testing, and keep quantizer calibration on training images.

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
checkpoint hash. Validation previews are in the training run's `validation_examples/`
artifacts; when resuming without a new best checkpoint, the existing previews remain
in that same MLflow run. After reconstruction review, use a validation evaluation for
compression experiments during development:

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
