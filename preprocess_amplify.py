"""
preprocess_amplify.py

Run this ONCE per dataset before training/evaluation. It:
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

Output layout (created by this script):

    <cache_root>/<sequence_name>/amplified/00000.jpg
    <cache_root>/<sequence_name>/motion/00001.npy   # motion map paired to frame t,
                                                      # computed from amplified (t-1, t)
                                                      # (frame 0 has no motion map)

Usage:
    python preprocess_amplify.py --data_root /path/to/MoCA-Mask --cache_root ./cache --fps 24
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from motion_amplification import amplify_video_in_windows
from optical_flow import compute_motion_map


def list_sequences(data_root: Path) -> list[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir()])


def load_frames(seq_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    frame_dir = seq_dir / "Frame"
    frame_files = sorted(os.listdir(frame_dir))
    frames = [cv2.imread(str(frame_dir / f)) for f in frame_files]
    return frames, frame_files


def process_sequence(seq_dir: Path, cache_root: Path, fps: float, window_size: int, overlap: int):
    frames, frame_files = load_frames(seq_dir)
    if len(frames) == 0:
        print(f"  [skip] no frames found in {seq_dir}")
        return

    print(f"  amplifying {len(frames)} frames ...")
    amplified = amplify_video_in_windows(
        frames, fps=fps, window_size=window_size, overlap=overlap,
        low_hz=0.2, high_hz=3.0, alpha=15.0, levels=4,
    )

    out_amp_dir = cache_root / seq_dir.name / "amplified"
    out_mot_dir = cache_root / seq_dir.name / "motion"
    out_amp_dir.mkdir(parents=True, exist_ok=True)
    out_mot_dir.mkdir(parents=True, exist_ok=True)

    for fname, amp_frame in zip(frame_files, amplified):
        cv2.imwrite(str(out_amp_dir / fname), amp_frame)

    print("  computing optical flow on amplified frames ...")
    for t in range(1, len(amplified)):
        motion_map = compute_motion_map(amplified[t - 1], amplified[t], backend="farneback")
        out_name = Path(frame_files[t]).stem + ".npy"
        np.save(str(out_mot_dir / out_name), motion_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    sequences = list_sequences(data_root)
    print(f"Found {len(sequences)} sequences under {data_root}")
    for seq_dir in sequences:
        print(f"Processing {seq_dir.name} ...")
        process_sequence(seq_dir, cache_root, args.fps, args.window_size, args.overlap)

    print(f"Done. Cached amplified frames + motion maps under {cache_root}")


if __name__ == "__main__":
    main()
