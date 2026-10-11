#!/usr/bin/env python3
"""
scripts/make_samples.py

Batch processes sample videos from samples_input/ using run_pipeline()
and writes precomputed multi-stream outputs to samples/<name>/ along with
a thumbnail.jpg and meta.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_demo import run_pipeline

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def make_samples(
    input_dir: str | Path = "samples_input",
    output_dir: str | Path = "samples",
    checkpoint_path: str | Path = "best_model.zip",
    alpha: float = 15.0,
    low_hz: float = 0.2,
    high_hz: float = 3.0,
    overwrite: bool = False,
) -> list[str]:
    in_dir = (REPO_ROOT / input_dir).resolve() if not Path(input_dir).is_absolute() else Path(input_dir)
    out_dir = (REPO_ROOT / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    ckpt = (REPO_ROOT / checkpoint_path).resolve() if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)

    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt}")

    video_files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])

    if not video_files:
        print(f"[*] No video files found in {in_dir}.")
        print(f"[*] Drop video clips (.mp4, .mov, etc.) into {in_dir} and re-run.")
        return []

    print(f"[*] Found {len(video_files)} video(s) in {in_dir}:")
    for vf in video_files:
        print(f"    - {vf.name}")

    processed_samples = []

    for idx, video_file in enumerate(video_files, 1):
        sample_name = video_file.stem.lower().replace(" ", "_")
        sample_target_dir = out_dir / sample_name

        print(f"\n[{idx}/{len(video_files)}] Processing: {video_file.name} -> {sample_target_dir.name}/")

        if sample_target_dir.exists() and (sample_target_dir / "raw.mp4").exists() and not overwrite:
            print(f"    [SKIP] Sample '{sample_name}' already computed. Ensuring metadata & thumbnail...")
            meta_path = sample_target_dir / "meta.json"
            if not meta_path.exists():
                title = sample_name.replace("_", " ").title()
                meta = {
                    "id": sample_name,
                    "title": title,
                    "description": f"Precomputed motion-amplified camouflage breaking sequence for {title}.",
                    "source_file": video_file.name,
                    "alpha": alpha,
                    "low_hz": low_hz,
                    "high_hz": high_hz,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                print(f"    [OK] Generated metadata: {meta_path.name}")
            thumb_path = sample_target_dir / "thumbnail.jpg"
            if not thumb_path.exists():
                cap = cv2.VideoCapture(str(sample_target_dir / "raw.mp4"))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    cv2.imwrite(str(thumb_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    print(f"    [OK] Generated thumbnail: {thumb_path.name}")
            processed_samples.append(sample_name)
            continue

        sample_target_dir.mkdir(parents=True, exist_ok=True)

        def progress_cb(stage: str, percent: float):
            print(f"\r    [{stage.upper()}] {percent:5.1f}%", end="", flush=True)

        t0 = time.time()
        try:
            # 1. Run pipeline
            run_pipeline(
                video_path=str(video_file),
                checkpoint_path=str(ckpt),
                out_dir=str(sample_target_dir),
                alpha=alpha,
                low_hz=low_hz,
                high_hz=high_hz,
                progress_cb=progress_cb,
                write_side_by_side=True,
            )
            print()

            # 2. Extract thumbnail.jpg from raw.mp4
            raw_video_path = sample_target_dir / "raw.mp4"
            thumb_path = sample_target_dir / "thumbnail.jpg"
            cap = cv2.VideoCapture(str(raw_video_path if raw_video_path.exists() else video_file))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                cv2.imwrite(str(thumb_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                print(f"    [OK] Generated thumbnail: {thumb_path.name}")
            else:
                print("    [WARN] Could not extract thumbnail frame.")

            # 3. Write meta.json
            title = sample_name.replace("_", " ").title()
            meta = {
                "id": sample_name,
                "title": title,
                "description": f"Precomputed motion-amplified camouflage breaking sequence for {title}.",
                "source_file": video_file.name,
                "alpha": alpha,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            meta_path = sample_target_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"    [OK] Generated metadata: {meta_path.name}")

            elapsed = time.time() - t0
            print(f"    [DONE] Completed '{sample_name}' in {elapsed:.1f}s")
            processed_samples.append(sample_name)

        except Exception as e:
            print(f"\n    [ERROR] Failed to process {video_file.name}: {e}")

    print(f"\n[+] Finished processing {len(processed_samples)}/{len(video_files)} sample(s).")
    return processed_samples


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate precomputed demo samples for Motion-Amplified Camouflage Breaking"
    )
    parser.add_argument("--input_dir", default="samples_input", help="Directory containing source video files")
    parser.add_argument("--output_dir", default="samples", help="Destination samples/ directory")
    parser.add_argument("--checkpoint", default="best_model.zip", help="Model checkpoint path")
    parser.add_argument("--alpha", type=float, default=15.0, help="Amplification factor alpha")
    parser.add_argument("--low_hz", type=float, default=0.2, help="Bandpass low cutoff frequency")
    parser.add_argument("--high_hz", type=float, default=3.0, help="Bandpass high cutoff frequency")
    parser.add_argument("--overwrite", action="store_true", help="Re-generate even if sample exists")

    args = parser.parse_args()

    make_samples(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
