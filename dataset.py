"""
dataset.py

PyTorch Dataset for MoCA-Mask-style video COD data. Reads:
  - the raw RGB frame (appearance stream input)
  - the precomputed motion-magnitude map from preprocess_amplify.py (motion stream input)
  - the ground-truth binary mask (only available every 5th frame in MoCA-Mask)

Only frames that have a ground-truth mask AND a valid motion map (i.e. not
frame 0 of a sequence, since it has no predecessor to compute flow against)
are used as training/evaluation samples.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MoCAMaskDataset(Dataset):
    """
    Expects, per sequence, this layout:
        <data_root>/<seq>/Frame/00000.jpg               (raw frames, for GT alignment)
        <data_root>/<seq>/GT/00000.png                   (ground-truth masks, sparse)
        <cache_root>/<seq>/amplified/00000.jpg           (from preprocess_amplify.py)
        <cache_root>/<seq>/motion/00001.npy              (from preprocess_amplify.py)

    Note: the RGB input to the appearance stream is the RAW frame (not
    amplified) — amplification is only used to make optical flow more
    reliable for the motion stream; the appearance stream should still see the
    true image so it isn't confused by amplification artifacts.
    """

    def __init__(self, data_root: str, cache_root: str, image_size: int = 352):
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.image_size = image_size
        self.samples = self._index_samples()

    def _index_samples(self) -> list[dict]:
        samples = []
        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_dir = seq_dir / "GT"
            frame_dir = seq_dir / "Frame"
            motion_dir = self.cache_root / seq_dir.name / "motion"
            if not gt_dir.exists() or not motion_dir.exists():
                continue

            for gt_file in sorted(os.listdir(gt_dir)):
                stem = Path(gt_file).stem
                frame_file = frame_dir / f"{stem}.jpg"
                motion_file = motion_dir / f"{stem}.npy"
                # skip frames with no predecessor motion map (e.g. frame 0)
                if frame_file.exists() and motion_file.exists():
                    samples.append(
                        {
                            "frame": frame_file,
                            "gt": gt_dir / gt_file,
                            "motion": motion_file,
                        }
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        rgb = cv2.imread(str(item["frame"]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

        motion = np.load(str(item["motion"]))
        motion = cv2.resize(motion, (self.image_size, self.image_size))
        motion = torch.from_numpy(motion).unsqueeze(0).float()

        mask = cv2.imread(str(item["gt"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return rgb, motion, mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python dataset.py <data_root> <cache_root>")
        sys.exit(0)

    ds = MoCAMaskDataset(sys.argv[1], sys.argv[2])
    print(f"Found {len(ds)} labeled samples with cached motion maps.")
    if len(ds) > 0:
        rgb, motion, mask = ds[0]
        print("rgb:", rgb.shape, "motion:", motion.shape, "mask:", mask.shape)
