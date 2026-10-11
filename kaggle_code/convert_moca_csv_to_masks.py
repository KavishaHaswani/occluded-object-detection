"""
convert_moca_csv_to_masks.py

MoCA's ground truth is NOT per-frame mask images — it's a single CSV file
exported from VGG Image Annotator (VIA), containing a bounding box (and a
motion-type label) per annotated frame. This script:

  1. Parses that CSV.
  2. For each annotated frame, rasterizes its bounding box into a binary mask
     PNG (same size as the real frame image).
  3. Writes those masks into a NEW folder tree that matches the same
     Frame/GT-per-sequence convention every other adapter in this project
     already expects (moca_mask_adapter), so MoCA slots into the same
     MultiSourceVCODDataset pipeline as everything else without any special
     casing at the dataset.py / preprocess_amplify.py level.
  4. Symlinks (does NOT copy) each sequence's raw JPEGImages folder into the
     new tree as "Frame/" — so the large frame images are never duplicated on
     disk, only the small generated mask PNGs are new files.

CSV format (VIA export), relevant columns:
    metadata_id, file_list, flags, temporal_coordinates, spatial_coordinates, metadata
  - file_list:            "/arabian_horn_viper/00000.jpg"  (seq name + frame filename)
  - spatial_coordinates:  "[2, x, y, width, height]"        (shape_id 2 = rectangle)
  - metadata:              {"1": "0"}  -> attribute "1" (motion) = option "0"
                            (0 = locomotion, 1 = subtle_motion, 2 = still)

Usage:
    python convert_moca_csv_to_masks.py \
        --moca_root /kaggle/input/datasets/ivanomelchenkoim11/moca-dataset/MoCA \
        --output_root /kaggle/working/moca_converted

Then use it exactly like any other dataset:
    from dataset import DatasetSource, moca_mask_adapter
    DatasetSource("moca", "/kaggle/working/moca_converted",
                  "/kaggle/working/cache/moca", moca_mask_adapter)
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


MOTION_LABELS = {"0": "locomotion", "1": "subtle_motion", "2": "still"}


def parse_via_csv(csv_path: Path) -> list[dict]:
    """
    Parses a VIA-exported CSV, skipping the '#'-prefixed documentation/header
    lines, and returns a list of {"seq": ..., "frame_file": ..., "box": (x,y,w,h),
    "motion_label": ...} dicts, one per annotated frame. Rows with no usable
    rectangle (empty spatial_coordinates, or a non-rectangle shape) are skipped.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue  # documentation / header comment lines
            if len(row) < 6:
                continue

            file_list, spatial_coordinates, metadata = row[1], row[4], row[5]

            # file_list looks like "/arabian_horn_viper/00000.jpg"
            file_list = file_list.strip().lstrip("/")
            if "/" not in file_list:
                continue
            seq_name, frame_file = file_list.split("/", 1)

            # spatial_coordinates looks like "[2,482.87,225.391,507.13,191.739]"
            try:
                coords = ast.literal_eval(spatial_coordinates)
            except (ValueError, SyntaxError):
                continue
            if not coords or coords[0] != 2 or len(coords) != 5:
                continue  # skip empty / non-rectangle annotations
            _, x, y, w, h = coords

            # metadata looks like {"1": "0"} -> motion label
            motion_label = "unknown"
            try:
                meta = ast.literal_eval(metadata.replace('""', '"'))
                motion_label = MOTION_LABELS.get(str(meta.get("1", "")), "unknown")
            except (ValueError, SyntaxError):
                pass

            rows.append(
                {
                    "seq": seq_name,
                    "frame_file": frame_file,
                    "box": (x, y, w, h),
                    "motion_label": motion_label,
                }
            )
    return rows


def make_mask(image_shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Rasterizes a single (x, y, w, h) rectangle into a binary mask of the given (H, W)."""
    h_img, w_img = image_shape
    x, y, w, h = box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(w_img, int(round(x + w))), min(h_img, int(round(y + h)))
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moca_root", type=str, required=True,
                         help="Path to the MoCA folder containing JPEGImages/ and Annotations/")
    parser.add_argument("--output_root", type=str, required=True,
                         help="Where to write the converted Frame/GT-style tree")
    parser.add_argument("--csv_name", type=str, default="annotations.csv")
    args = parser.parse_args()

    moca_root = Path(args.moca_root)
    output_root = Path(args.output_root)
    csv_path = moca_root / "Annotations" / args.csv_name
    jpeg_root = moca_root / "JPEGImages"

    print(f"Parsing {csv_path} ...")
    rows = parse_via_csv(csv_path)
    print(f"Parsed {len(rows)} annotated frames across "
          f"{len(set(r['seq'] for r in rows))} sequences.")

    motion_counts = defaultdict(int)
    for r in rows:
        motion_counts[r["motion_label"]] += 1
    print("Motion-label breakdown (useful for picking clips that stress-test "
          "motion-based detection):")
    for label, count in sorted(motion_counts.items()):
        print(f"  {label}: {count}")

    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)

    n_masks_written, n_skipped = 0, 0
    for seq_name, seq_rows in by_seq.items():
        seq_out = output_root / seq_name
        gt_dir = seq_out / "GT"
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Symlink the real frame folder in as "Frame" -- no copying of the
        # (large) original JPEGs, only the new small mask PNGs are written.
        frame_link = seq_out / "Frame"
        real_frame_dir = jpeg_root / seq_name
        if not frame_link.exists():
            try:
                os.symlink(real_frame_dir.resolve(), frame_link)
            except OSError as e:
                print(f"  [warn] could not symlink {real_frame_dir} -> {frame_link}: {e}")
                continue

        for r in seq_rows:
            frame_path = real_frame_dir / r["frame_file"]
            if not frame_path.exists():
                n_skipped += 1
                continue
            img = cv2.imread(str(frame_path))
            if img is None:
                n_skipped += 1
                continue

            mask = make_mask(img.shape[:2], r["box"])
            out_name = Path(r["frame_file"]).stem + ".png"
            cv2.imwrite(str(gt_dir / out_name), mask)
            n_masks_written += 1

    print(f"\nDone. Wrote {n_masks_written} mask PNGs "
          f"({n_skipped} rows skipped due to missing frames) under {output_root}")
    print(f"Now usable exactly like any other dataset via moca_mask_adapter:")
    print(f'  DatasetSource("moca", "{output_root}", "<cache_root>/moca", moca_mask_adapter)')


if __name__ == "__main__":
    main()
