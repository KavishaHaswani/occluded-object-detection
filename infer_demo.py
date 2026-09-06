"""
infer_demo.py

End-to-end demo: takes a raw input video, runs the full pipeline (amplify ->
optical flow -> two-stream model), and writes an output video with:
  - Left panel:  original frame with a "baseline" mask overlay (appearance-only,
                 i.e. the same model but fed a zeroed-out motion map - this
                 simulates what a plain single-frame detector would show).
  - Right panel: original frame with the full motion-amplified pipeline's mask
                 overlay.

This side-by-side is the actual hackathon demo deliverable described earlier.

Usage:
    python infer_demo.py --video input.mp4 --checkpoint ./checkpoints/best_model.pth \
                          --output demo_output.mp4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from motion_amplification import amplify_video_in_windows
from optical_flow import compute_motion_map
from model import TwoStreamCOD

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def preprocess_rgb(frame_bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()


def overlay_mask(frame_bgr: np.ndarray, prob_map: np.ndarray, color=(0, 0, 255), alpha=0.5) -> np.ndarray:
    prob_resized = cv2.resize(prob_map, (frame_bgr.shape[1], frame_bgr.shape[0]))
    mask = (prob_resized > 0.5).astype(np.uint8)

    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    color_layer[:, :] = color
    overlay = np.where(mask[..., None] == 1, cv2.addWeighted(frame_bgr, 1 - alpha, color_layer, alpha, 0), frame_bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="demo_output.mp4")
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading video ...")
    frames, fps = read_video(args.video)
    print(f"{len(frames)} frames @ {fps:.1f} fps")

    print("Amplifying motion (sliding windows) ...")
    amplified = amplify_video_in_windows(
        frames, fps=fps, window_size=args.window_size, overlap=args.overlap,
        low_hz=0.2, high_hz=3.0, alpha=15.0, levels=4,
    )

    print("Loading model ...")
    model = TwoStreamCOD(base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h)
    )

    print("Running inference frame-by-frame ...")
    with torch.no_grad():
        for t in range(1, len(frames)):
            rgb_tensor = preprocess_rgb(frames[t], args.image_size).to(device)

            # Baseline (appearance-only): zero out the motion stream.
            zero_motion = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
            baseline_logits = model(rgb_tensor, zero_motion)
            baseline_prob = torch.sigmoid(baseline_logits)[0, 0].cpu().numpy()

            # Full pipeline: motion map from amplified frame pair (t-1, t).
            motion_map = compute_motion_map(amplified[t - 1], amplified[t], backend="farneback")
            motion_map_resized = cv2.resize(motion_map, (args.image_size, args.image_size))
            motion_tensor = torch.from_numpy(motion_map_resized).unsqueeze(0).unsqueeze(0).float().to(device)

            full_logits = model(rgb_tensor, motion_tensor)
            full_prob = torch.sigmoid(full_logits)[0, 0].cpu().numpy()

            left = overlay_mask(frames[t], baseline_prob, color=(0, 165, 255))  # orange = baseline
            right = overlay_mask(frames[t], full_prob, color=(0, 0, 255))       # red = full pipeline

            cv2.putText(left, "Appearance-only (baseline)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(right, "Motion-amplified (ours)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            combined = np.hstack([left, right])
            writer.write(combined)

    writer.release()
    print(f"Done. Demo video written to {args.output}")


if __name__ == "__main__":
    main()
