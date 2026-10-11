"""
preprocess_amplify.py

Run this ONCE per dataset (or once per multiple datasets, via --data_roots)
before training/evaluation. It:
  1. Reads every frame of a video sequence.
  2. Runs Eulerian motion amplification in sliding windows (motion_amplification.py).
  3. Computes optical flow between every consecutive pair of AMPLIFIED frames
     (optical_flow.py) and saves the normalized motion-magnitude map.

Doing this once and caching to disk (rather than inside the DataLoader) matters
because amplification needs a whole window of frames and is too slow to redo on
every __getitem__ call during training.

Expected MoCA-Mask style input layout (adjust `list_sequences` if yours differs):

    <data_root>/<sequence_name>/Frame/00000.jpg
    <data_root>/<sequence_name>/Frame/00001.jpg
    ...
    <data_root>/<sequence_name>/GT/00000.png      # ground-truth mask, every 5th frame
    <data_root>/<sequence_name>/GT/00005.png
    ...

Output layout (created by this script), one subfolder per dataset name so
multiple datasets' caches never collide:

    <cache_root>/<dataset_name>/<sequence_name>/amplified/00000.jpg
    <cache_root>/<dataset_name>/<sequence_name>/motion/00001.npy

Usage (single dataset):
    python preprocess_amplify.py --data_root /kaggle/input/moca-mask/train \
                                  --cache_root /kaggle/working/cache --fps 24

Usage (multiple datasets in one run - e.g. all 4 Kaggle-attached datasets):
    python preprocess_amplify.py \
        --data_roots moca_mask=/kaggle/input/moca-mask/train \
                     cad=/kaggle/input/cad-dataset \
                     msvcod=/kaggle/input/msvcod/train \
                     yuv20k=/kaggle/input/yuv20k/train \
        --cache_root /kaggle/working/cache --fps 24

    This produces:
        /kaggle/working/cache/moca_mask/<seq>/motion/*.npy
        /kaggle/working/cache/cad/<seq>/motion/*.npy
        /kaggle/working/cache/msvcod/<seq>/motion/*.npy
        /kaggle/working/cache/yuv20k/<seq>/motion/*.npy
    which is exactly the layout dataset.py's MultiSourceVCODDataset expects
    when you point each DatasetSource's cache_root at <cache_root>/<name>.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from motion_amplification import amplify_video_in_windows
from optical_flow import compute_motion_map


FRAME_SUBDIR_CANDIDATES = ["Frame", "Frames", "frames", "images", "Images", "img",
                            "JPEGImages", "Imgs", "imgs"]


def _is_junk_file(filename: str) -> bool:
    """Filters out macOS AppleDouble metadata files (e.g. '._image.jpg') and
    .DS_Store, which commonly appear in datasets zipped on a Mac."""
    return filename.startswith("._") or filename == ".DS_Store"


def find_frame_dir(seq_dir: Path) -> Path:
    """Auto-detects which subfolder inside a sequence holds the raw frames,
    since datasets differ in naming convention (Frame/, images/, Imgs/, ...)."""
    for candidate in FRAME_SUBDIR_CANDIDATES:
        candidate_path = seq_dir / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path
    raise FileNotFoundError(
        f"Could not find a frame folder under {seq_dir}. "
        f"Tried: {FRAME_SUBDIR_CANDIDATES}. "
        f"If your dataset uses a different name, add it to FRAME_SUBDIR_CANDIDATES above."
    )


def list_sequences(data_root: Path) -> list[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir()])


def load_frames(seq_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    frame_dir = find_frame_dir(seq_dir)
    frame_files = sorted(f for f in os.listdir(frame_dir) if not _is_junk_file(f))
    frames = []
    valid_files = []
    for f in frame_files:
        img = cv2.imread(str(frame_dir / f))
        if img is None:
            print(f"  [warn] could not read {frame_dir / f}, skipping")
            continue
        frames.append(img)
        valid_files.append(f)
    return frames, valid_files


def process_sequence(seq_dir: Path, cache_root: Path, fps: float, window_size: int, overlap: int,
                     motion_size: int = 0, save_amplified: bool = False):
    try:
        frames, frame_files = load_frames(seq_dir)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return

    if len(frames) == 0:
        print(f"  [skip] no frames found in {seq_dir}")
        return

    print(f"  amplifying {len(frames)} frames ...")
    amplified = amplify_video_in_windows(
        frames, fps=fps, window_size=window_size, overlap=overlap,
        low_hz=0.2, high_hz=3.0, alpha=15.0, levels=4,
    )

    out_mot_dir = cache_root / seq_dir.name / "motion"
    out_mot_dir.mkdir(parents=True, exist_ok=True)

    if save_amplified:
        out_amp_dir = cache_root / seq_dir.name / "amplified"
        out_amp_dir.mkdir(parents=True, exist_ok=True)
        for fname, amp_frame in zip(frame_files, amplified):
            cv2.imwrite(str(out_amp_dir / fname), amp_frame)

    print("  computing optical flow on amplified frames ...")
    for t in range(1, len(amplified)):
        motion_map = compute_motion_map(amplified[t - 1], amplified[t], backend="farneback")
        if motion_size and motion_size > 0:
            motion_map = cv2.resize(motion_map, (motion_size, motion_size))
        motion_map = motion_map.astype(np.float16)
        out_name = Path(frame_files[t]).stem + ".npy"
        out_path = out_mot_dir / out_name
        try:
            np.save(str(out_path), motion_map)
        except OSError:
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            raise


def process_dataset(data_root: Path, cache_root: Path, dataset_name: str, fps: float, window_size: int,
                    overlap: int, motion_size: int = 0, save_amplified: bool = False):
    dataset_cache_root = cache_root / dataset_name
    sequences = list_sequences(data_root)
    print(f"[{dataset_name}] found {len(sequences)} sequences under {data_root}")
    for seq_dir in sequences:
        print(f"[{dataset_name}] processing {seq_dir.name} ...")
        process_sequence(seq_dir, dataset_cache_root, fps, window_size, overlap, motion_size, save_amplified)


def parse_named_roots(items: list[str]) -> dict[str, str]:
    """Parses ['name1=path1', 'name2=path2', ...] into a dict."""
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--data_roots entries must be name=path, got: {item}")
        name, path = item.split("=", 1)
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None,
                         help="Single dataset root (uses folder name as dataset_name).")
    parser.add_argument("--data_roots", type=str, nargs="+", default=None,
                         help="Multiple datasets as name=path pairs, e.g. moca_mask=/kaggle/input/moca-mask cad=/kaggle/input/cad")
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--motion_size", type=int, default=0,
                         help="Resize motion maps to this square size before saving (0 = keep original).")
    parser.add_argument("--save_amplified", action="store_true",
                         help="Also cache amplified frames (not used by dataset.py; off by default to save disk).")
    args = parser.parse_args()

    if not args.data_root and not args.data_roots:
        raise ValueError("Pass either --data_root (single dataset) or --data_roots (multiple, name=path pairs).")

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.data_roots:
        named_roots = parse_named_roots(args.data_roots)
    else:
        data_root = Path(args.data_root)
        named_roots = {data_root.name: str(data_root)}

    for dataset_name, path in named_roots.items():
        process_dataset(Path(path), cache_root, dataset_name, args.fps, args.window_size, args.overlap,
                        args.motion_size, args.save_amplified)

    print(f"\nDone. Cached amplified frames + motion maps under {cache_root}/<dataset_name>/...")
    print("Point each DatasetSource's cache_root in dataset.py at one of these subfolders.")


if __name__ == "__main__":
    main()

