"""
infer_demo.py

End-to-end demo and backend pipeline runner for motion-amplified camouflage breaking:
Takes a raw input video, runs the full pipeline (amplify -> optical flow -> two-stream model),
and produces separate synchronized video streams ready for web frontend playback, plus an
optional side-by-side demo video.

Outputs generated per video (all frames 1..N-1, synchronized in FPS and resolution):
  - raw.mp4: original input video without overlays
  - baseline.mp4: appearance-only baseline overlay (zeroed motion map), without text
  - amplified.mp4: full motion-amplified pipeline mask overlay, without text
  - amplified_frames.mp4: Eulerian-amplified frames without overlay
  - flow.mp4: motion-magnitude heatmap (COLORMAP_JET) normalized across the entire video
  - demo_output.mp4: side-by-side comparison video (baseline vs ours with text labels)

All outputs are re-encoded to H.264 (libx264, yuv420p, +faststart) for browser compatibility.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import cv2
import numpy as np
import torch

from motion_amplification import amplify_video_in_windows
from optical_flow import compute_motion_map, farneback_flow, flow_to_magnitude
from model import TwoStreamCOD

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Module-level model cache to avoid reloading weights across repeated calls.
_MODEL_CACHE: dict[tuple[str, int, str], TwoStreamCOD] = {}


def get_cached_model(checkpoint_path: str, base_ch: int, device: torch.device) -> TwoStreamCOD:
    """Retrieves a cached model instance or loads and caches it."""
    cache_key = (os.path.abspath(checkpoint_path), base_ch, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = TwoStreamCOD(base_ch=base_ch).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


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


def reencode_to_h264(video_path: Path, ffmpeg_bin: str) -> bool:
    """Re-encodes an mp4v video to browser-compatible H.264 (yuv420p, faststart)."""
    if not video_path.exists():
        return False

    tmp_path = video_path.with_name(f"{video_path.stem}_h264_tmp.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        os.replace(tmp_path, video_path)
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        stderr_msg = e.stderr.decode("utf-8", errors="replace") if hasattr(e, "stderr") and e.stderr else str(e)
        print(f"Warning: Failed to re-encode '{video_path}' with ffmpeg: {stderr_msg}. Keeping original file.")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False


def run_pipeline(
    video_path: str,
    checkpoint_path: str,
    out_dir: str,
    alpha: float = 15.0,
    low_hz: float = 0.2,
    high_hz: float = 3.0,
    image_size: int = 352,
    base_ch: int = 32,
    window_size: int = 60,
    overlap: int = 30,
    progress_cb: Callable[[str, float], None] | None = None,
    write_side_by_side: bool = True,
) -> dict:
    """
    Runs the full motion-amplified camouflage breaking pipeline and writes separate
    synchronized browser-playable videos to out_dir.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Reading stage
    if progress_cb:
        progress_cb("reading", 0.0)
    frames, fps = read_video(str(video_path))
    if len(frames) < 2:
        raise ValueError(f"Video '{video_path}' contains {len(frames)} frames; at least 2 frames are required.")
    if progress_cb:
        progress_cb("reading", 100.0)

    # 2. Amplifying stage
    if progress_cb:
        progress_cb("amplifying", 0.0)
    amplified = amplify_video_in_windows(
        frames,
        fps=fps,
        window_size=window_size,
        overlap=overlap,
        low_hz=low_hz,
        high_hz=high_hz,
        alpha=alpha,
        levels=4,
    )
    if progress_cb:
        progress_cb("amplifying", 100.0)

    # 3. Model setup (using cached instance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_cached_model(checkpoint_path=checkpoint_path, base_ch=base_ch, device=device)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    raw_path = out_dir_path / "raw.mp4"
    baseline_path = out_dir_path / "baseline.mp4"
    amplified_path = out_dir_path / "amplified.mp4"
    amplified_frames_path = out_dir_path / "amplified_frames.mp4"
    flow_path = out_dir_path / "flow.mp4"
    demo_path = out_dir_path / "demo_output.mp4"

    raw_writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))
    baseline_writer = cv2.VideoWriter(str(baseline_path), fourcc, fps, (w, h))
    amplified_writer = cv2.VideoWriter(str(amplified_path), fourcc, fps, (w, h))
    amplified_frames_writer = cv2.VideoWriter(str(amplified_frames_path), fourcc, fps, (w, h))
    demo_writer = cv2.VideoWriter(str(demo_path), fourcc, fps, (w * 2, h)) if write_side_by_side else None

    total_infer_frames = len(frames) - 1
    raw_mags: list[np.ndarray] = []

    # 4. Inference stage
    if progress_cb:
        progress_cb("inference", 0.0)

    with torch.no_grad():
        for i, t in enumerate(range(1, len(frames))):
            rgb_tensor = preprocess_rgb(frames[t], image_size).to(device)

            # Baseline (appearance-only): zero out the motion stream
            zero_motion = torch.zeros(1, 1, image_size, image_size, device=device)
            baseline_logits = model(rgb_tensor, zero_motion)
            baseline_prob = torch.sigmoid(baseline_logits)[0, 0].cpu().numpy()

            # Optical flow between amplified consecutive frames (t-1, t)
            flow = farneback_flow(amplified[t - 1], amplified[t])
            raw_mag = flow_to_magnitude(flow, normalize=False)
            raw_mags.append(raw_mag)

            # Normalized motion map passed to model (identical to compute_motion_map)
            motion_map = flow_to_magnitude(flow, normalize=True)
            motion_map_resized = cv2.resize(motion_map, (image_size, image_size))
            motion_tensor = torch.from_numpy(motion_map_resized).unsqueeze(0).unsqueeze(0).float().to(device)

            # Full pipeline forward pass
            full_logits = model(rgb_tensor, motion_tensor)
            full_prob = torch.sigmoid(full_logits)[0, 0].cpu().numpy()

            # Clean overlays without text labels
            baseline_frame = overlay_mask(frames[t], baseline_prob, color=(0, 165, 255))  # orange
            amplified_frame = overlay_mask(frames[t], full_prob, color=(0, 0, 255))      # red

            raw_writer.write(frames[t])
            baseline_writer.write(baseline_frame)
            amplified_writer.write(amplified_frame)
            amplified_frames_writer.write(amplified[t])

            # Optional side-by-side with labels
            if demo_writer is not None:
                left_labeled = baseline_frame.copy()
                right_labeled = amplified_frame.copy()
                cv2.putText(left_labeled, "Appearance-only (baseline)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(right_labeled, "Motion-amplified (ours)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                combined = np.hstack([left_labeled, right_labeled])
                demo_writer.write(combined)

            if progress_cb:
                progress_cb("inference", float((i + 1) / total_infer_frames) * 100.0)

    raw_writer.release()
    baseline_writer.release()
    amplified_writer.release()
    amplified_frames_writer.release()
    if demo_writer is not None:
        demo_writer.release()

    # 5. Write flow heatmap video normalized across the whole video
    flow_writer = cv2.VideoWriter(str(flow_path), fourcc, fps, (w, h))
    global_max = max((float(m.max()) for m in raw_mags), default=1.0)
    if global_max < 1e-6:
        global_max = 1.0

    for mag in raw_mags:
        norm_mag = np.clip((mag / global_max) * 255.0, 0, 255).astype(np.uint8)
        flow_heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
        flow_writer.write(flow_heatmap)
    flow_writer.release()

    # 6. Re-encode videos to browser-compatible H.264 via ffmpeg
    videos_to_reencode = [
        raw_path,
        baseline_path,
        amplified_path,
        amplified_frames_path,
        flow_path,
    ]
    if write_side_by_side:
        videos_to_reencode.append(demo_path)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("Warning: ffmpeg not found in PATH. Keeping original mp4v files (may not play in all web browsers).")
    else:
        for vid in videos_to_reencode:
            reencode_to_h264(vid, ffmpeg_bin)

    return {
        "raw": str(raw_path.resolve()),
        "baseline": str(baseline_path.resolve()),
        "amplified": str(amplified_path.resolve()),
        "amplified_frames": str(amplified_frames_path.resolve()),
        "flow": str(flow_path.resolve()),
        "demo_output": str(demo_path.resolve()) if write_side_by_side else None,
        "outputs": {
            "raw": str(raw_path.resolve()),
            "baseline": str(baseline_path.resolve()),
            "amplified": str(amplified_path.resolve()),
            "amplified_frames": str(amplified_frames_path.resolve()),
            "flow": str(flow_path.resolve()),
            "demo_output": str(demo_path.resolve()) if write_side_by_side else None,
        },
        "frame_count": total_infer_frames,
        "fps": float(fps),
        "width": int(w),
        "height": int(h),
    }


def main():
    parser = argparse.ArgumentParser(description="Motion-amplified camouflage breaking inference demo")
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory for output videos")
    parser.add_argument("--output", type=str, default="demo_output.mp4", help="Combined side-by-side demo output path")
    parser.add_argument("--alpha", type=float, default=15.0, help="Eulerian motion amplification factor")
    parser.add_argument("--low_hz", type=float, default=0.2, help="Temporal bandpass filter low cutoff (Hz)")
    parser.add_argument("--high_hz", type=float, default=3.0, help="Temporal bandpass filter high cutoff (Hz)")
    parser.add_argument("--image_size", type=int, default=352, help="Image size for model input")
    parser.add_argument("--base_ch", type=int, default=32, help="Base channel count for TwoStreamCOD")
    parser.add_argument("--window_size", type=int, default=60, help="Sliding window size for motion amplification")
    parser.add_argument("--overlap", type=int, default=30, help="Sliding window overlap size")
    args = parser.parse_args()

    def cli_progress(stage: str, percent: float):
        if stage == "inference":
            print(f"\rInference progress: {percent:.1f}%", end="", flush=True)
            if percent >= 100.0:
                print()
        elif percent == 0.0:
            print(f"Starting stage: {stage} ...")
        elif percent >= 100.0:
            print(f"Completed stage: {stage}")

    res = run_pipeline(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        alpha=args.alpha,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        image_size=args.image_size,
        base_ch=args.base_ch,
        window_size=args.window_size,
        overlap=args.overlap,
        progress_cb=cli_progress,
        write_side_by_side=True,
    )

    if args.output and res.get("demo_output"):
        out_target = Path(args.output).resolve()
        demo_src = Path(res["demo_output"]).resolve()
        if out_target != demo_src:
            out_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(demo_src, out_target)
            print(f"Combined demo output copied to: {out_target}")

    print(f"\nDone. Outputs saved to '{args.out_dir}':")
    for name, path in res.get("outputs", {}).items():
        if path:
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
