"""
preprocess_amplify.py

Run this ONCE per dataset (or once per multiple datasets, via --data_roots)
before training/evaluation. It:
  1. Reads every frame of a video sequence.
  2. Runs Eulerian motion amplification in sliding windows (motion_amplification.py),
     UNLESS --no_amplify is passed.
  3. Computes optical flow between every consecutive pair of frames (amplified
     in the normal path, raw in --no_amplify mode) and saves the normalized
     motion-magnitude map.

Doing this once and caching to disk (rather than inside the DataLoader) matters
because amplification needs a whole window of frames and is too slow to redo on
every __getitem__ call during training.

ABLATION: --no_amplify builds the "without motion amplification" cache used by
the raw-cache training run. Always point such a run at a SEPARATE cache_root
(e.g. /kaggle/working/cache_raw) so raw and amplified motion maps never mix.

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
from concurrent.futures import ProcessPoolExecutor
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
                     motion_size: int = 0, save_amplified: bool = False, no_amplify: bool = False,
                     alpha: float = 15.0, low_hz: float = 0.2, high_hz: float = 3.0, levels: int = 4,
                     brightness_only: bool = False, coarse_only: bool = False, median_sub: bool = False,
                     norm_mode: str = "max"):
    try:
        frames, frame_files = load_frames(seq_dir)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return

    if len(frames) == 0:
        print(f"  [skip] no frames found in {seq_dir}")
        return

    if no_amplify:
        print(f"  [raw] skipping Eulerian amplification, using {len(frames)} raw frames ...")
        processed = frames
    else:
        print(f"  amplifying {len(frames)} frames (alpha={alpha}, [{low_hz}, {high_hz}] Hz, coarse_only={coarse_only}, brightness_only={brightness_only}) ...")
        processed = amplify_video_in_windows(
            frames, fps=fps, window_size=window_size, overlap=overlap,
            low_hz=low_hz, high_hz=high_hz, alpha=alpha, levels=levels,
            brightness_only=brightness_only, coarse_only=coarse_only,
        )

    out_mot_dir = cache_root / seq_dir.name / "motion"
    out_mot_dir.mkdir(parents=True, exist_ok=True)

    if save_amplified and not no_amplify:
        out_amp_dir = cache_root / seq_dir.name / "amplified"
        out_amp_dir.mkdir(parents=True, exist_ok=True)
        for fname, amp_frame in zip(frame_files, processed):
            cv2.imwrite(str(out_amp_dir / fname), amp_frame)

    domain = "raw frames" if no_amplify else "amplified frames"
    print(f"  computing optical flow on {domain} (median_sub={median_sub}, norm_mode={norm_mode}) ...")
    for t in range(1, len(processed)):
        motion_map = compute_motion_map(
            processed[t - 1], processed[t], backend="farneback",
            median_sub=median_sub, norm_mode=norm_mode,
        )
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


def _process_sequence_star(kwargs):
    return process_sequence(**kwargs)


def process_dataset(data_root: Path, cache_root: Path, dataset_name: str, fps: float, window_size: int,
                    overlap: int, motion_size: int = 0, save_amplified: bool = False,
                    no_amplify: bool = False, num_workers: int = 1,
                    alpha: float = 15.0, low_hz: float = 0.2, high_hz: float = 3.0, levels: int = 4,
                    brightness_only: bool = False, coarse_only: bool = False, median_sub: bool = False,
                    norm_mode: str = "max"):
    dataset_cache_root = cache_root / dataset_name
    sequences = list_sequences(data_root)
    print(f"[{dataset_name}] found {len(sequences)} sequences under {data_root}")
    tasks = [
        dict(seq_dir=seq_dir, cache_root=dataset_cache_root, fps=fps, window_size=window_size,
             overlap=overlap, motion_size=motion_size, save_amplified=save_amplified,
             no_amplify=no_amplify, alpha=alpha, low_hz=low_hz, high_hz=high_hz, levels=levels,
             brightness_only=brightness_only, coarse_only=coarse_only, median_sub=median_sub,
             norm_mode=norm_mode)
        for seq_dir in sequences
    ]
    if num_workers and num_workers > 1:
        print(f"[{dataset_name}] processing with {num_workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for _ in ex.map(_process_sequence_star, tasks):
                pass
    else:
        for task in tasks:
            print(f"[{dataset_name}] processing {task['seq_dir'].name} ...")
            process_sequence(**task)


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
    import json
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
    parser.add_argument("--no_amplify", action="store_true",
                         help="Ablation: skip Eulerian amplification and compute optical flow directly on the "
                              "raw frames (use a separate --cache_root so raw/amplified maps never mix).")
    parser.add_argument("--alpha", type=float, default=15.0, help="Eulerian amplification factor alpha")
    parser.add_argument("--low_hz", type=float, default=0.2, help="Bandpass lower cutoff frequency in Hz")
    parser.add_argument("--high_hz", type=float, default=3.0, help="Bandpass higher cutoff frequency in Hz")
    parser.add_argument("--levels", type=int, default=4, help="Laplacian pyramid decomposition levels")
    parser.add_argument("--brightness_only", action="store_true", help="Amplify Y channel only in YUV color space")
    parser.add_argument("--coarse_only", action="store_true", help="Amplify only coarse pyramid levels (leave level 0 unamplified)")
    parser.add_argument("--median_sub", action="store_true", help="Subtract spatial median flow vector before magnitude calculation")
    parser.add_argument("--norm_mode", choices=["max", "percentile"], default="max", help="Flow normalization mode")
    parser.add_argument("--num_workers", type=int, default=1,
                         help="Parallel worker processes across sequences (per dataset). 1 = sequential. "
                              "Use os.cpu_count() on Kaggle to parallelize the amplification.")
    args = parser.parse_args()

    if not args.data_root and not args.data_roots:
        raise ValueError("Pass either --data_root (single dataset) or --data_roots (multiple, name=path pairs).")

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    config = {
        "no_amplify": args.no_amplify,
        "fps": args.fps,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "motion_size": args.motion_size,
        "alpha": args.alpha,
        "low_hz": args.low_hz,
        "high_hz": args.high_hz,
        "levels": args.levels,
        "brightness_only": args.brightness_only,
        "coarse_only": args.coarse_only,
        "median_sub": args.median_sub,
        "norm_mode": args.norm_mode,
    }
    (cache_root / "preprocess_config.json").write_text(json.dumps(config, indent=2))

    if args.data_roots:
        named_roots = parse_named_roots(args.data_roots)
    else:
        data_root = Path(args.data_root)
        named_roots = {data_root.name: str(data_root)}

    for dataset_name, path in named_roots.items():
        process_dataset(Path(path), cache_root, dataset_name, args.fps, args.window_size, args.overlap,
                        args.motion_size, args.save_amplified, args.no_amplify, args.num_workers,
                        alpha=args.alpha, low_hz=args.low_hz, high_hz=args.high_hz, levels=args.levels,
                        brightness_only=args.brightness_only, coarse_only=args.coarse_only,
                        median_sub=args.median_sub, norm_mode=args.norm_mode)

    mode = "raw (unamplified)" if args.no_amplify else "amplified"
    print(f"\nDone. Cached {mode} motion maps under {cache_root}/<dataset_name>/...")
    print("Point each DatasetSource's cache_root in dataset.py at one of these subfolders.")


if __name__ == "__main__":
    main()

