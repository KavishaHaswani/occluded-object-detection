"""
dataset.py

PyTorch Dataset for VCOD data — supports merging MULTIPLE datasets (e.g. four
separate Kaggle datasets: MoCA-Mask, CAD, MSVCOD, etc.) without physically
copying/merging their files. Each dataset keeps its own folder; this class
scans all of them and builds one combined sample list.

Different datasets often use different folder-naming conventions, so each
dataset is described by an "adapter" that tells this class how to find the
Frame/GT folders inside it. A default adapter matching the MoCA-Mask layout
(Frame/, GT/) is provided; write a small custom adapter for any dataset that
doesn't match this layout (see `make_custom_adapter` for a quick factory).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, NamedTuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_split_sequences(split_dir: Path, split: str, source_name: str) -> set:
    """Loads the sequence-level split file <split_dir>/<source_name>_<split>.json
    written by make_split.py and returns it as a set of sequence names.

    Raises if the file is missing: silently training on everything when a split
    was requested would hide train/val/test leakage.
    """
    path = Path(split_dir) / f"{source_name}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run make_split.py first (it writes "
            f"<source>_<split>.json files), or drop --split_dir to train on all data."
        )
    seqs = set(json.loads(path.read_text()))
    if not seqs:
        raise ValueError(f"Split file {path} is empty - refusing to build a 0-sample dataset.")
    return seqs


class DatasetSource(NamedTuple):
    """One attached dataset: its raw root, its cache root, and how to read its sequences."""
    name: str                       # human-readable label, e.g. "moca_mask"
    data_root: str                  # e.g. "/kaggle/input/moca-mask"
    cache_root: str                 # e.g. "/kaggle/working/cache/moca_mask"
    adapter: Callable[[Path], "list[dict]"]  # see moca_mask_adapter() below


def _is_junk_file(filename: str) -> bool:
    """Filters out macOS AppleDouble metadata files (e.g. '._image.png') and
    .DS_Store, which commonly appear in datasets zipped on a Mac and will
    otherwise crash cv2.imread with a confusing error."""
    return filename.startswith("._") or filename == ".DS_Store"


def moca_mask_adapter(seq_dir: Path) -> list[dict]:
    """
    Default adapter for MoCA-Mask-style layout:
        <seq_dir>/Frame/00000.jpg, 00001.jpg, ...
        <seq_dir>/GT/00000.png, 00005.png, ...   (sparse, every 5th frame)
    Returns a list of {"frame": path, "gt": path, "stem": "00000"} dicts, one
    per frame that HAS a ground-truth mask.
    """
    frame_dir = seq_dir / "Frame"
    gt_dir = seq_dir / "GT"
    if not frame_dir.exists() or not gt_dir.exists():
        return []

    entries = []
    for gt_file in sorted(os.listdir(gt_dir)):
        if _is_junk_file(gt_file):
            continue
        stem = Path(gt_file).stem
        # frame extension can vary (.jpg/.png) across datasets; try both.
        for ext in (".jpg", ".png", ".jpeg"):
            frame_file = frame_dir / f"{stem}{ext}"
            if frame_file.exists():
                entries.append({"frame": frame_file, "gt": gt_dir / gt_file, "stem": stem})
                break
    return entries


def cad_adapter(seq_dir: Path) -> list[dict]:
    """
    Adapter for the Camouflaged Animal Dataset (CAD) layout, e.g.:
        <seq_dir>/frames/chameleon_001.png, chameleon_002.png, ...
        <seq_dir>/groundtruth/001_gt.png, 006_gt.png, ...   (every 5th frame)

    Frame and GT filenames use DIFFERENT naming schemes (sequence-name prefix
    vs. "_gt" suffix), so they're matched by their shared zero-padded frame
    number rather than by an identical stem. The returned "stem" is the FRAME
    file's own stem (e.g. "chameleon_001"), not the GT-derived number, because
    preprocess_amplify.py names cached motion .npy files after the raw frame
    filename's stem — this keeps the two scripts' naming consistent.
    """
    frame_dir = seq_dir / "frames"
    gt_dir = seq_dir / "groundtruth"
    if not frame_dir.exists() or not gt_dir.exists():
        return []

    seq_name = seq_dir.name
    entries = []
    for gt_file in sorted(os.listdir(gt_dir)):
        if _is_junk_file(gt_file):
            continue
        gt_stem = Path(gt_file).stem  # e.g. "001_gt"
        if not gt_stem.endswith("_gt"):
            continue
        frame_number = gt_stem[: -len("_gt")]  # "001"
        for ext in (".png", ".jpg", ".jpeg"):
            frame_file = frame_dir / f"{seq_name}_{frame_number}{ext}"
            if frame_file.exists():
                entries.append(
                    {"frame": frame_file, "gt": gt_dir / gt_file, "stem": frame_file.stem}
                )
                break
    return entries


def make_custom_adapter(frame_subdir: str, gt_subdir: str, frame_ext: str = ".jpg") -> Callable:
    """
    Quick factory for datasets that use Frame/GT-style layout (SAME filename
    stem in both folders) but with different folder names — e.g. YUV20K's
    "Imgs"/"GT" pair, where Imgs/00255.png and GT/00255.png share a stem.

    Usage:
        adapter = make_custom_adapter("Imgs", "GT", frame_ext=".png")
        source = DatasetSource("yuv20k", "/kaggle/input/.../YUV20K", "/kaggle/working/cache/yuv20k", adapter)
    """
    def adapter(seq_dir: Path) -> list[dict]:
        frame_dir = seq_dir / frame_subdir
        gt_dir = seq_dir / gt_subdir
        if not frame_dir.exists() or not gt_dir.exists():
            return []
        entries = []
        for gt_file in sorted(os.listdir(gt_dir)):
            if _is_junk_file(gt_file):
                continue
            stem = Path(gt_file).stem
            frame_file = frame_dir / f"{stem}{frame_ext}"
            if frame_file.exists():
                entries.append({"frame": frame_file, "gt": gt_dir / gt_file, "stem": stem})
        return entries
    return adapter


class MultiSourceVCODDataset(Dataset):
    """
    Merges multiple VCOD datasets into a single PyTorch Dataset. Each source's
    raw data stays exactly where it is (e.g. read-only /kaggle/input/...);
    only the precomputed motion maps (from preprocess_amplify.py) live under
    each source's own cache_root, so nothing large gets physically copied or
    duplicated — this class only builds an index (a list of file paths) that
    points into each source's own folder.
    """

    def __init__(self, sources: list[DatasetSource], image_size: int = 352,
                 split_dir: str | None = None, split: str | None = None):
        self.image_size = image_size
        self.split_dir = split_dir
        self.split = split
        self.samples = []  # list of dicts, each tagged with its source name

        for source in sources:
            data_root = Path(source.data_root)
            cache_root = Path(source.cache_root)
            if not data_root.exists():
                print(f"  [warn] data_root not found, skipping source '{source.name}': {data_root}")
                continue

            allowed_seqs = None
            if split_dir is not None:
                if not split:
                    raise ValueError("split_dir was given but split is empty (expected train/val/test).")
                allowed_seqs = load_split_sequences(Path(split_dir), split, source.name)

            n_seqs_kept = 0
            n_seqs_skipped = 0
            n_before = len(self.samples)
            for seq_dir in sorted(data_root.iterdir()):
                if not seq_dir.is_dir():
                    continue
                if allowed_seqs is not None and seq_dir.name not in allowed_seqs:
                    n_seqs_skipped += 1
                    continue
                motion_dir = cache_root / seq_dir.name / "motion"
                if not motion_dir.exists():
                    continue  # this sequence hasn't been preprocessed yet

                entries = source.adapter(seq_dir)
                n_entries_used = 0
                for e in entries:
                    motion_file = motion_dir / f"{e['stem']}.npy"
                    if motion_file.exists():
                        self.samples.append(
                            {
                                "frame": e["frame"],
                                "gt": e["gt"],
                                "motion": motion_file,
                                "source": source.name,
                                "sequence": seq_dir.name,
                            }
                        )
                        n_entries_used += 1
                if n_entries_used:
                    n_seqs_kept += 1
            label = f"split={split}" if split_dir is not None else "split=all"
            print(
                f"  [{source.name}] {label}: {n_seqs_kept} sequences with cached motion used, "
                f"{n_seqs_skipped} outside split, {len(self.samples) - n_before} samples"
            )

        print(f"Total merged samples across {len(sources)} sources: {len(self.samples)}")

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
        if motion.ndim == 2:
            h_m, w_m = motion.shape
        elif motion.ndim == 3:
            h_m, w_m = motion.shape[:2]
        else:
            h_m, w_m = self.image_size, self.image_size
        # Resize to target if shapes differ
        if (h_m != self.image_size) or (w_m != self.image_size):
            motion = cv2.resize(motion, (self.image_size, self.image_size))
        motion = motion.astype(np.float32)
        if motion.ndim == 2:
            motion = motion
        elif motion.ndim == 3:
            motion = motion[:, :, 0] if motion.shape[2] > 1 else motion[:, :, 0]
        motion = torch.from_numpy(motion).unsqueeze(0).float()

        mask = cv2.imread(str(item["gt"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if mask.max() <= 2:
            mask = (mask > 0).astype(np.float32)
        else:
            mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return rgb, motion, mask

    def source_breakdown(self) -> dict:
        """Handy for checking class balance across datasets, e.g. before training."""
        counts = {}
        for s in self.samples:
            counts[s["source"]] = counts.get(s["source"], 0) + 1
        return counts


# Backwards-compatible single-dataset wrapper (same interface as before, used
# by train.py / eval.py's --data_root/--cache_root single-path arguments).
class MoCAMaskDataset(MultiSourceVCODDataset):
    def __init__(self, data_root: str, cache_root: str, image_size: int = 352):
        source = DatasetSource("single", data_root, cache_root, moca_mask_adapter)
        super().__init__([source], image_size=image_size)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset.py <data_root> <cache_root>  [single-dataset test]")
        sys.exit(0)

    ds = MoCAMaskDataset(sys.argv[1], sys.argv[2])
    print(f"Found {len(ds)} labeled samples with cached motion maps.")
    if len(ds) > 0:
        rgb, motion, mask = ds[0]
        print("rgb:", rgb.shape, "motion:", motion.shape, "mask:", mask.shape)
