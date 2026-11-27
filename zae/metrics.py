import torch
import torch.nn.functional as F


def round_rgb(x):
    return x.clamp(0, 1).mul(255).round().div(255)


def psnr(a, b):
    mse = (a - b).square().flatten(1).mean(1)
    return -10 * torch.log10(mse)


def ssim(a, b):
    """RGB channel-mean SSIM: Gaussian 11x11, sigma=1.5, valid region, range=1."""
    if min(a.shape[-2:]) < 11:
        raise ValueError("SSIM needs at least 11x11 pixels")
    coords = torch.arange(11, dtype=a.dtype, device=a.device) - 5
    g = torch.exp(-coords.square() / (2 * 1.5 ** 2))
    g = g / g.sum()
    channels = a.shape[1]
    wx = g.reshape(1, 1, 1, 11).repeat(channels, 1, 1, 1)
    wy = g.reshape(1, 1, 11, 1).repeat(channels, 1, 1, 1)

    def filt(x):
        return F.conv2d(F.conv2d(x, wx, groups=channels), wy, groups=channels)

    ma, mb = filt(a), filt(b)
    va, vb, cov = filt(a * a) - ma.square(), filt(b * b) - mb.square(), filt(a * b) - ma * mb
    score = ((2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (ma.square() + mb.square() + 0.01 ** 2) * (va + vb + 0.03 ** 2))
    return score.flatten(1).mean(1)


def encode_padded(model, x):
    h, w = x.shape[-2:]
    return model.encode(F.pad(x, (0, (-w) % model.factor, 0, (-h) % model.factor), mode="replicate"))
