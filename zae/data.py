import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def center_crop(x, size):
    h, w = x.shape[-2:]
    if min(h, w) < size:
        raise ValueError(f"Crop {size} exceeds image size {h}x{w}")
    top, left = (h - size) // 2, (w - size) // 2
    return x[:, top:top + size, left:left + size]


def manifest(cfg):
    root = Path(cfg.data.root).expanduser().resolve()
    train = sorted(p for d in cfg.data.train_dirs for p in (root / d).glob("*.png"))
    test = sorted((root / cfg.data.test_dir).glob("*.png"))
    if len(train) != cfg.data.expected_train_images or len(test) != cfg.data.expected_test_images:
        raise ValueError(f"Expected {cfg.data.expected_train_images}/{cfg.data.expected_test_images} "
                         f"train/test images, found {len(train)}/{len(test)} under {root}")
    if len({p.name for p in train}) != len(train) or {p.name for p in train} & {p.name for p in test}:
        raise ValueError("Duplicate training image names or train/test overlap")
    count = cfg.data.validation_images
    if not 0 < count < len(train):
        raise ValueError("validation_images must leave nonempty training and validation sets")
    held = set(random.Random(cfg.data.split_seed).sample(train, count))
    splits = {"train": [p for p in train if p not in held], "validation": sorted(held), "test": test}
    result = {key: [p.relative_to(root).as_posix() for p in paths] for key, paths in splits.items()}
    # Hash contents, not machine-specific absolute paths; same data on another PC has the same ID.
    hashes = {}
    for p in train + test:
        hashes[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    result["sha256"] = hashes
    result["dataset_id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def split_paths(cfg, saved_manifest, split):
    root = Path(cfg.data.root).expanduser().resolve()
    return [root / p for p in saved_manifest[split]]


class TrainingCrops(Dataset):
    """Index-seeded samples allow the same crop stream after checkpoint resume."""
    def __init__(self, files, size, count, seed, flip):
        self.files, self.size, self.count, self.seed, self.flip = files, size, count, seed, flip

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)
        x = load_rgb(self.files[rng.randrange(len(self.files))])
        h, w = x.shape[-2:]
        if min(h, w) < self.size:
            raise ValueError("Training image smaller than crop")
        top, left = rng.randrange(h - self.size + 1), rng.randrange(w - self.size + 1)
        x = x[:, top:top + self.size, left:left + self.size]
        return x.flip(-1) if self.flip and rng.random() < 0.5 else x
