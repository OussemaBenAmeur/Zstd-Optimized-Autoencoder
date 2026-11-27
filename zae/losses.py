"""Differentiable int8 compressibility proxies; actual Zstd sizes are measured separately.

There is no gradient through Zstd. A triangular soft histogram approximates symbol
entropy, while adjacent CHW symbols are encouraged to repeat. Neither proxy is a
prediction or bound on the Zstd file size. Decoder architecture/weights are unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from zae.codec import quantize, serialize


class ZstdRateLoss(nn.Module):
    def __init__(self, scales, qmax, lambda_rate, spatial_weight, overflow_weight):
        super().__init__()
        self.register_buffer("scales", scales.detach().clone())
        self.qmax = qmax
        self.lambda_rate = lambda_rate
        self.spatial_weight = spatial_weight
        self.overflow_weight = overflow_weight

    def quantized_latent(self, z):
        # Exact int8 forward values, identity surrogate gradient through round/clamp.
        z = z.float()
        hard = quantize(z, self.scales, self.qmax).to(z.dtype) * self.scales
        return hard.detach() + (z - z.detach())

    def rate_terms(self, z, pixels_per_image):
        z = z.float()
        units = z / self.scales
        values = units.clamp(-self.qmax, self.qmax).flatten(1)
        symbols_per_pixel = values.shape[1] / pixels_per_image
        # Two-bin interpolation uses O(N + bins) storage rather than O(N * bins).
        positions = values + self.qmax
        lower = positions.floor().long()
        fraction = positions - lower
        histogram = values.new_zeros((values.shape[0], 2 * self.qmax + 1))
        histogram = histogram.scatter_add(1, lower, 1 - fraction)
        histogram = histogram.scatter_add(1, (lower + 1).clamp_max(2 * self.qmax), fraction)
        probabilities = histogram / values.shape[1]
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log2()).sum(1).mean()
        # CHW C-order is the actual byte order passed to Zstd. Include row/channel boundaries.
        differences = values[:, 1:] - values[:, :-1]
        spatial = (torch.log2(1 + differences.abs()).mean() if differences.numel()
                   else values.sum() * 0)
        # Keep a gradient for values beyond int8 range, where the clamped proxies saturate.
        overflow = F.relu(units.abs() - self.qmax).square().mean()
        entropy_bpp, spatial_bpp = entropy * symbols_per_pixel, spatial * symbols_per_pixel
        rate = entropy_bpp + self.spatial_weight * spatial_bpp + self.overflow_weight * overflow
        return {"rate_proxy_bpp": rate, "entropy_bpp": entropy_bpp,
                "spatial_proxy_bpp": spatial_bpp, "overflow": overflow}

    def forward(self, model, x):
        z = model.encode(x)
        # Only convolutions inherit v3's bf16 autocast. Quantization and the loss must
        # retain float32 precision, especially histogram probabilities and small rates.
        with torch.autocast(z.device.type, enabled=False):
            z = z.float()
            quantized = self.quantized_latent(z)
        reconstruction = model.decode(quantized)
        with torch.autocast(z.device.type, enabled=False):
            mse = F.mse_loss(reconstruction.float(), x.float())
            terms = self.rate_terms(z, x.shape[-2] * x.shape[-1])
            loss = mse + self.lambda_rate * terms["rate_proxy_bpp"]
        return loss, {"mse": mse, "loss": loss, **terms}


def zstd_file_sizes(z, scales, qmax, image_height, image_width, zstd_config):
    """Measure complete per-image files with the evaluation codec, including metadata.

    Checkpoint hashes are fixed-width and stored uncompressed in the header, so a
    placeholder gives exactly the same size as an eventual real checkpoint hash.
    """
    arrays = quantize(z, scales, qmax).detach().cpu().numpy()
    metadata = {"image_height": image_height, "image_width": image_width,
                "quantized": True, "scales": scales.flatten().tolist(),
                "checkpoint_sha256": "0" * 64}
    return [len(serialize(array, metadata, zstd_config)) for array in arrays]
