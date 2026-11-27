"""Static figures for training and evaluation runs.

Every figure is regenerated from files saved in the run's output directory, so plots can be
rebuilt later without retraining:

    python -m zae.plots training   outputs/<experiment>/<timestamp>
    python -m zae.plots evaluation outputs/<experiment>/<timestamp>
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical order (blue, orange, aqua, violet); identity is also carried by legends.
BLUE, ORANGE, AQUA, VIOLET = "#257de0", "#f47230", "#00b894", "#6a4ce2"
TEXT, MUTED, GRID, REFERENCE = "#191919", "#666666", "#e6e6e6", "#9c9c9c"
BRANCH_COLORS = {"float32": BLUE, "int8": ORANGE}
ERROR_MAP = LinearSegmentedColormap.from_list("error", ["#ffffff", "#84b9f3", "#2370cd", "#0e3d7e"])


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": TEXT,
    })


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _empty(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _series(rows, key):
    pairs = [(r["step"], r[key]) for r in rows if r.get(key) not in (None, "")]
    return [p[0] for p in pairs], [float(p[1]) for p in pairs]


# ----------------------------------------------------------------------------------- training

def training_curves(history, performance, path, title="", early_stopping=None, gradient_clip=None):
    """3x3 small multiples; each panel has a single y-axis and one unit."""
    _style()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), constrained_layout=True)
    (loss_ax, psnr_ax, ssim_ax), (lr_ax, grad_ax, patience_ax), (latent_ax, speed_ax, wait_ax) = axes

    steps, mse = _series(history, "train_mse")
    if steps:
        loss_ax.plot(steps, mse, color=BLUE)
        loss_ax.set_yscale("log")
        loss_ax.set_title("Training loss (MSE, log scale)")
        loss_ax.set_ylabel("MSE")
    else:
        _empty(loss_ax, "no training records yet")

    val_steps, val_psnr = _series(history, "val_psnr")
    if steps:
        psnr_ax.plot(steps, [-10 * math.log10(v) for v in mse], color=BLUE,
                     label="train (from window MSE, 256² crops, unrounded)")
    if val_steps:
        psnr_ax.plot(val_steps, val_psnr, color=ORANGE, marker="o", label="validation (512² crops, RGB8)")
        lo_steps, lo = _series(history, "val_psnr_min")
        hi_steps, hi = _series(history, "val_psnr_max")
        if lo_steps and lo_steps == hi_steps:
            psnr_ax.fill_between(lo_steps, lo, hi, color=ORANGE, alpha=0.12, linewidth=0,
                                 label="validation min–max across images")
        best = int(np.argmax(val_psnr))
        psnr_ax.annotate(f"best {val_psnr[best]:.2f} dB\nstep {val_steps[best]}",
                         (val_steps[best], val_psnr[best]), textcoords="offset points", xytext=(-10, -30),
                         ha="right", color=TEXT, fontsize=8,
                         arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        psnr_ax.scatter([val_steps[best]], [val_psnr[best]], s=60, facecolor=ORANGE,
                        edgecolor="white", linewidth=2, zorder=5)
    if steps or val_steps:
        psnr_ax.set_title("PSNR")
        psnr_ax.set_ylabel("dB")
        psnr_ax.legend(loc="lower right")
    else:
        _empty(psnr_ax, "no PSNR records yet")

    s_steps, s_val = _series(history, "val_ssim")
    if s_steps:
        ssim_ax.plot(s_steps, s_val, color=ORANGE, marker="o")
        ssim_ax.set_title("Validation SSIM")
        ssim_ax.ticklabel_format(axis="y", useOffset=False)
    else:
        _empty(ssim_ax, "no validation yet")

    lr_steps, lr = _series(history, "learning_rate")
    if lr_steps:
        lr_ax.plot(lr_steps, lr, color=BLUE)
        lr_ax.set_title("Learning rate (logged steps)")
        lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        _empty(lr_ax, "no training records yet")

    g_steps, g_mean = _series(history, "gradient_norm_mean")
    if not g_steps:
        g_steps, g_mean = _series(history, "gradient_norm")
    gm_steps, g_max = _series(history, "gradient_norm_max")
    if g_steps:
        grad_ax.plot(g_steps, g_mean, color=BLUE, label="window mean")
        if gm_steps:
            grad_ax.plot(gm_steps, g_max, color=AQUA, label="window max")
        if gradient_clip:
            grad_ax.axhline(gradient_clip, color=REFERENCE, linewidth=1)
            grad_ax.annotate(f"clip {gradient_clip:g}", (1, gradient_clip), xycoords=("axes fraction", "data"),
                             ha="right", va="bottom", color=MUTED, fontsize=8)
        grad_ax.set_title("Gradient norm before clipping")
        grad_ax.set_yscale("log")
        grad_ax.legend(loc="lower left")
    else:
        _empty(grad_ax, "no training records yet")

    b_steps, bad = _series(history, "early_stopping_bad_checks")
    if b_steps:
        patience_ax.bar(b_steps, bad, width=max(1, 0.6 * (b_steps[1] - b_steps[0]) if len(b_steps) > 1 else 1),
                        color=VIOLET)
        if early_stopping and early_stopping.get("enabled"):
            patience_ax.axhline(early_stopping["patience"], color=REFERENCE, linewidth=1)
            patience_ax.annotate(f"stop at {early_stopping['patience']}", (1, early_stopping["patience"]),
                                 xycoords=("axes fraction", "data"), ha="right", va="bottom",
                                 color=MUTED, fontsize=8)
        patience_ax.set_title("Early stopping: checks without significant gain")
        patience_ax.yaxis.get_major_locator().set_params(integer=True)
    else:
        _empty(patience_ax, "no validation yet")

    l_steps, l_std = _series(history, "latent_std")
    _, l_p999 = _series(history, "latent_abs_p999")
    if l_steps:
        latent_ax.plot(l_steps, l_std, color=BLUE, marker="o", label="std")
        if l_p999:
            latent_ax.plot(l_steps, l_p999, color=ORANGE, marker="o", label="|z| 99.9th percentile")
        latent_ax.set_title("Validation latent scale (float32)")
        latent_ax.set_ylabel("latent units")
        latent_ax.legend(loc="best")
        latent_ax.ticklabel_format(axis="y", useOffset=False)
    else:
        _empty(latent_ax, "no latent statistics yet")

    p_steps, images = _series(performance, "images_per_second")
    if p_steps:
        speed_ax.plot(p_steps, images, color=BLUE)
        speed_ax.set_title("Throughput")
        speed_ax.set_ylabel("training crops / s")
        speed_ax.set_ylim(bottom=0)
    else:
        _empty(speed_ax, "no performance records")

    w_steps, wait = _series(performance, "data_wait_fraction")
    if w_steps:
        wait_ax.plot(w_steps, [100 * v for v in wait], color=BLUE)
        wait_ax.set_title("Time waiting for data loader")
        wait_ax.set_ylabel("% of wall time")
        wait_ax.set_ylim(0, 100)
    else:
        _empty(wait_ax, "no performance records")

    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("optimizer step")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def reconstruction_grid(examples, path, title=""):
    """examples: list of dicts with original/reconstruction HWC float arrays and metrics."""
    _style()
    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.3 * rows), squeeze=False, constrained_layout=True)
    worst = max(float(np.abs(e["original"] - e["reconstruction"]).mean(-1).max()) for e in examples)
    vmax = max(worst, 1 / 255)
    for row, example in zip(axes, examples):
        error = np.abs(example["original"] - example["reconstruction"]).mean(-1)
        row[0].imshow(example["original"])
        row[0].set_title(f"{example['name']} — original")
        row[1].imshow(example["reconstruction"])
        row[1].set_title(f"reconstruction — {example['psnr']:.2f} dB, SSIM {example['ssim']:.4f}")
        image = row[2].imshow(error, cmap=ERROR_MAP, vmin=0, vmax=vmax)
        row[2].set_title("absolute error (RGB mean)")
        fig.colorbar(image, ax=row[2], fraction=0.046, pad=0.02).outline.set_visible(False)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


# --------------------------------------------------------------------------------- evaluation

def _by_branch(rows):
    result = {}
    for row in rows:
        result.setdefault(row["branch"], []).append(row)
    return result


def rate_distortion(rows, path, title=""):
    _style()
    groups = _by_branch(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, label in [(axes[0], "psnr_db", "PSNR (dB)"), (axes[1], "ssim", "SSIM")]:
        for branch, selected in groups.items():
            color = BRANCH_COLORS.get(branch, VIOLET)
            bpp = [float(r["bpp_after_zstd"]) for r in selected]
            quality = [float(r[metric]) for r in selected]
            ax.scatter(bpp, quality, s=22, color=color, alpha=0.55, edgecolor="white", linewidth=0.6,
                       label=f"{branch} latent + Zstd (per image)")
            pixels = sum(float(r["height"]) * float(r["width"]) for r in selected)
            total_bpp = 8 * sum(float(r["zstd_file_bytes"]) for r in selected) / pixels
            ax.scatter([total_bpp], [np.mean(quality)], s=110, marker="D", color=color, edgecolor="white",
                       linewidth=2, zorder=5, label=f"{branch} dataset ({total_bpp:.2f} bpp, {np.mean(quality):.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("bits per original pixel after Zstd (log scale)")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split(' ')[0]} versus compressed size")
        ax.legend(loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def storage_breakdown(summary, path, title=""):
    """Dataset-level bits per pixel for every storage condition."""
    _style()
    bars = []
    any_branch = next(iter(summary.values()))
    pixels = 8 * any_branch["raw_rgb_bytes_total"] / 24
    bars.append(("raw RGB8", 8 * any_branch["raw_rgb_bytes_total"] / pixels, REFERENCE))
    bars.append(("original PNG", 8 * any_branch["original_png_bytes_total"] / pixels, REFERENCE))
    for branch, values in summary.items():
        color = BRANCH_COLORS.get(branch, VIOLET)
        bars.append((f"{branch} latent file", values["bpp_before_zstd"], color))
        bars.append((f"{branch} latent file + Zstd", values["bpp_after_zstd"], color))
        if "order0_entropy_bytes_total" in values:
            bars.append((f"{branch} payload order-0 entropy", 8 * values["order0_entropy_bytes_total"] / pixels,
                         color))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(bars) + 1.5), constrained_layout=True)
    labels = [b[0] for b in bars][::-1]
    values = [b[1] for b in bars][::-1]
    colors = [b[2] for b in bars][::-1]
    alphas = [0.45 if "entropy" in label else 1.0 for label in labels]
    y = np.arange(len(bars))
    for yi, value, color, alpha in zip(y, values, colors, alphas):
        ax.barh(yi, value, color=color, alpha=alpha, height=0.62)
        ax.annotate(f"{value:.3f} bpp", (value, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", color=TEXT, fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("bits per original pixel (dataset total bytes / total pixels, log scale)")
    ax.set_title(title or "Storage cost per pixel")
    return _save(fig, path)


def zstd_efficiency(rows, path, title=""):
    _style()
    groups = _by_branch(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ratio_ax, entropy_ax = axes
    for branch, selected in groups.items():
        color = BRANCH_COLORS.get(branch, VIOLET)
        ratio = [float(r["zstd_size_reduction_factor"]) for r in selected]
        ratio_ax.hist(ratio, bins=25, color=color, alpha=0.6, label=f"{branch} (mean {np.mean(ratio):.3f}×)")
        if selected and "order0_entropy_bytes" in selected[0]:
            bound = [float(r["order0_entropy_bytes"]) / 1024 for r in selected]
            actual = [float(r["zstd_file_bytes"]) / 1024 for r in selected]
            entropy_ax.scatter(bound, actual, s=22, color=color, alpha=0.7, edgecolor="white", linewidth=0.6,
                               label=branch)
    ratio_ax.axvline(1, color=REFERENCE, linewidth=1)
    ratio_ax.set_title("Zstd size reduction per image (file bytes / Zstd bytes)")
    ratio_ax.set_xlabel("reduction factor (> 1 means Zstd helped)")
    ratio_ax.set_ylabel("images")
    ratio_ax.legend()
    if entropy_ax.has_data():
        lo, hi = entropy_ax.get_xlim()[0], max(entropy_ax.get_xlim()[1], entropy_ax.get_ylim()[1])
        entropy_ax.plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1, label="Zstd = order-0 entropy")
        entropy_ax.set_title("Zstd file size versus byte-level order-0 entropy")
        entropy_ax.set_xlabel("order-0 entropy of latent bytes (KiB)")
        entropy_ax.set_ylabel("actual Zstd file (KiB)")
        entropy_ax.legend()
    else:
        _empty(entropy_ax, "entropy columns not recorded")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def quality_distribution(rows, path, title=""):
    _style()
    groups = _by_branch(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    dist_ax, drop_ax = axes
    for branch, selected in groups.items():
        values = [float(r["psnr_db"]) for r in selected]
        dist_ax.hist(values, bins=25, color=BRANCH_COLORS.get(branch, VIOLET), alpha=0.6,
                     label=f"{branch} (mean {np.mean(values):.2f} dB)")
    dist_ax.set_title("PSNR across test images")
    dist_ax.set_xlabel("PSNR (dB)")
    dist_ax.set_ylabel("images")
    dist_ax.legend()
    if {"float32", "int8"} <= groups.keys():
        float_psnr = {r["image"]: float(r["psnr_db"]) for r in groups["float32"]}
        drops = sorted((float_psnr[r["image"]] - float(r["psnr_db"]), r["image"]) for r in groups["int8"])
        drop_ax.bar(range(len(drops)), [d for d, _ in drops], color=ORANGE, width=0.8)
        drop_ax.set_title("PSNR lost by int8 quantization, per image (sorted)")
        drop_ax.set_xlabel("images, sorted by loss")
        drop_ax.set_ylabel("float32 PSNR − int8 PSNR (dB)")
        drop_ax.set_xticks([])
        worst_drop, worst_name = drops[-1]
        drop_ax.annotate(f"{worst_name}: {worst_drop:.3f} dB", (len(drops) - 1, worst_drop),
                         xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                         color=TEXT, fontsize=8)
    else:
        _empty(drop_ax, "needs both float32 and int8 branches")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def latent_statistics(stats, path, title=""):
    """stats: dict saved as latent_stats.json by evaluate.py."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    hist_ax, entropy_ax, scale_ax, clip_ax = axes.flat
    if "int8_symbol_counts" in stats:
        counts = np.asarray(stats["int8_symbol_counts"], dtype=float)
        symbols = np.arange(-128, 128)
        nonzero = counts > 0
        hist_ax.bar(symbols[nonzero], counts[nonzero] / counts.sum(), width=1.0, color=ORANGE)
        hist_ax.set_yscale("log")
        hist_ax.set_title(f"int8 symbol distribution (all channels; entropy "
                          f"{stats['int8_entropy_bits']:.3f} bits/symbol)")
        hist_ax.set_xlabel("quantized latent value")
        hist_ax.set_ylabel("probability (log scale)")
        channel_entropy = stats["int8_channel_entropy_bits"]
        order = np.argsort(channel_entropy)[::-1]
        entropy_ax.bar(range(len(order)), [channel_entropy[i] for i in order], color=ORANGE, width=0.8)
        entropy_ax.axhline(stats["int8_entropy_bits"], color=REFERENCE, linewidth=1)
        entropy_ax.annotate("pooled entropy", (1, stats["int8_entropy_bits"]),
                            xycoords=("axes fraction", "data"), ha="right", va="bottom", color=MUTED, fontsize=8)
        entropy_ax.set_title("Per-channel int8 entropy (sorted)")
        entropy_ax.set_xlabel("latent channel (sorted by entropy)")
        entropy_ax.set_ylabel("bits / symbol")
        entropy_ax.set_xticks([])
        channels = np.arange(len(stats["scales"]))
        scale_ax.bar(channels, stats["scales"], color=BLUE, width=0.8)
        scale_ax.set_title("Quantization step per channel (calibrated on training crops)")
        scale_ax.set_xlabel("latent channel")
        scale_ax.set_ylabel("step size (latent units)")
        clip_ax.bar(channels, [100 * v for v in stats["int8_channel_clipped_fraction"]], color=BLUE, width=0.8)
        clip_ax.set_title("Values clipped by int8 range, per channel (evaluated images)")
        clip_ax.set_xlabel("latent channel")
        clip_ax.set_ylabel("% of values")
    else:
        for ax in axes.flat:
            _empty(ax, "quantization disabled")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    return _save(fig, path)


def evaluation_samples(samples, path, title=""):
    """samples: dicts with name, original and per-branch reconstructions (HWC float arrays)."""
    _style()
    branches = list(samples[0]["reconstructions"])
    cols = 1 + 2 * len(branches)
    fig, axes = plt.subplots(len(samples), cols, figsize=(3.6 * cols, 3.8 * len(samples)),
                             squeeze=False, constrained_layout=True)
    vmax = max(1 / 255, max(float(np.abs(s["original"] - r["image"]).mean(-1).max())
                            for s in samples for r in s["reconstructions"].values()))
    for row, sample in zip(axes, samples):
        row[0].imshow(sample["original"])
        row[0].set_title(f"{sample['name']} — original crop")
        for index, branch in enumerate(branches):
            reconstruction = sample["reconstructions"][branch]
            row[1 + 2 * index].imshow(reconstruction["image"])
            row[1 + 2 * index].set_title(f"{branch}: {reconstruction['psnr']:.2f} dB, "
                                         f"{reconstruction['bpp']:.2f} bpp")
            error = np.abs(sample["original"] - reconstruction["image"]).mean(-1)
            row[2 + 2 * index].imshow(error, cmap=ERROR_MAP, vmin=0, vmax=vmax)
            row[2 + 2 * index].set_title(f"{branch} absolute error")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if title:
        fig.suptitle(title + f"  (error colour scale 0–{vmax:.3f})", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=TEXT)
    return _save(fig, path)


def evaluation_figures(rows, summary, stats, directory, title=""):
    directory = Path(directory)
    paths = [rate_distortion(rows, directory / "rate_distortion.png", title),
             storage_breakdown(summary, directory / "storage_breakdown.png", title),
             zstd_efficiency(rows, directory / "zstd_efficiency.png", title),
             quality_distribution(rows, directory / "quality_distribution.png", title)]
    if stats:
        paths.append(latent_statistics(stats, directory / "latent_statistics.png", title))
    return paths


# ------------------------------------------------------------------------------------ command line

def _read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main(argv):
    if len(argv) != 2 or argv[0] not in ("training", "evaluation"):
        raise SystemExit("usage: python -m zae.plots {training|evaluation} <output_dir>")
    kind, directory = argv[0], Path(argv[1])
    if kind == "training":
        history = json.loads((directory / "history.json").read_text())
        performance_file = directory / "performance.json"
        performance = json.loads(performance_file.read_text()) if performance_file.exists() else []
        summary_file = directory / "training_summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        print(training_curves(history, performance, directory / "plots" / "training_curves.png",
                              title=f"Training — {directory.name}",
                              early_stopping=summary.get("early_stopping_config"),
                              gradient_clip=summary.get("gradient_clip")))
    else:
        rows = _read_csv(directory / "per_image.csv")
        summary = json.loads((directory / "summary.json").read_text())
        stats_file = directory / "latent_stats.json"
        stats = json.loads(stats_file.read_text()) if stats_file.exists() else None
        for path in evaluation_figures(rows, summary, stats, directory / "plots",
                                       f"Evaluation — {directory.name}"):
            print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
