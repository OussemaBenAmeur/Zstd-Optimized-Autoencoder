"""Versioned latent files; sizes include the header and decoder side information."""
import json
import struct

import numpy as np
import torch
import zstandard as zstd


PREFIX = struct.Struct("<4sBI")  # magic, compressed flag, JSON header length


def compressor(cfg):
    return zstd.ZstdCompressor(level=cfg.level, threads=cfg.threads,
                              write_checksum=cfg.checksum)


def serialize(array, metadata, zstd_config=None):
    dtype = np.dtype("<f4") if array.dtype.kind == "f" else np.dtype("i1")
    array = np.ascontiguousarray(array, dtype=dtype)
    header = {**metadata, "dtype": dtype.str, "shape": list(array.shape), "layout": "CHW"}
    info = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload = array.tobytes(order="C")
    compressed = zstd_config is not None
    if compressed:
        payload = compressor(zstd_config).compress(payload)
    return PREFIX.pack(b"ZAE1", compressed, len(info)) + info + payload


def deserialize(blob):
    magic, compressed, length = PREFIX.unpack_from(blob)
    if magic != b"ZAE1" or compressed not in (0, 1):
        raise ValueError("Invalid latent file")
    end = PREFIX.size + length
    header = json.loads(blob[PREFIX.size:end])
    payload = blob[end:]
    if compressed:
        payload = zstd.ZstdDecompressor().decompress(payload)
    dtype = np.dtype(header["dtype"])
    if dtype.str not in ("<f4", "|i1") or len(header["shape"]) != 3:
        raise ValueError("Unsupported latent dtype/shape")
    expected = int(np.prod(header["shape"])) * dtype.itemsize
    if len(payload) != expected:
        raise ValueError("Latent payload length mismatch")
    return np.frombuffer(payload, dtype=dtype).reshape(header["shape"]).copy(), header


def quantize(z, scales, qmax=127):
    return (z / scales).round().clamp(-qmax, qmax).to(torch.int8)


def latent_tensor(array, metadata, device):
    z = torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0)
    if metadata["quantized"]:
        scales = torch.tensor(metadata["scales"], device=device).reshape(1, -1, 1, 1)
        z = z * scales
    return z
