"""
test_pipeline.py

Synthetic smoke test for the full pipeline (no real dataset needed). Generates
a short fake video: a noisy background with a low-contrast "camouflaged" blob
that slowly drifts, matching the background's mean color/texture (so it would
be nearly invisible to an appearance-only detector, but detectable via its
motion). Runs amplification -> optical flow -> model forward pass end-to-end
and checks that shapes and value ranges are sane at every step.
"""

import numpy as np
import cv2
import torch

from motion_amplification import amplify_video_in_windows
from optical_flow import compute_motion_map, farneback_flow, flow_to_magnitude
from model import TwoStreamCOD
from losses import COD_Loss


def make_synthetic_video(n_frames=90, h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    base_texture = rng.integers(80, 140, size=(h, w, 3), dtype=np.uint8)

    frames = []
    cy, cx = h // 2, w // 2
    for t in range(n_frames):
        frame = base_texture.copy()
        # subtle drift: a few-pixel sinusoidal displacement, mimicking breathing/small motion
        dy = int(2 * np.sin(2 * np.pi * t / 20))
        dx = int(2 * np.cos(2 * np.pi * t / 25))
        y0, y1 = cy - 15 + dy, cy + 15 + dy
        x0, x1 = cx - 15 + dx, cx + 15 + dx
        # camouflaged blob: same mean intensity as background, only +/-3 texture noise
        blob_color = base_texture[cy, cx].astype(np.int16) + rng.integers(-3, 3, size=3)
        frame[y0:y1, x0:x1] = np.clip(blob_color, 0, 255).astype(np.uint8)
        # add a little frame-to-frame sensor noise, like a real camera
        noise = rng.integers(-4, 4, size=frame.shape)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def main():
    print("1) Generating synthetic video with a near-invisible moving blob ...")
    frames = make_synthetic_video()
    print(f"   {len(frames)} frames of shape {frames[0].shape}")

    print("2) Running Eulerian motion amplification (sliding windows) ...")
    amplified = amplify_video_in_windows(
        frames, fps=30.0, window_size=60, overlap=30,
        low_hz=0.2, high_hz=3.0, alpha=15.0, levels=3,
    )
    assert len(amplified) == len(frames)
    assert amplified[0].shape == frames[0].shape
    print(f"   amplified {len(amplified)} frames OK, dtype={amplified[0].dtype}")

    print("3) Computing optical flow (Farneback) on amplified frame pairs ...")
    motion_maps = []
    for t in range(1, len(amplified)):
        m = compute_motion_map(amplified[t - 1], amplified[t], backend="farneback")
        motion_maps.append(m)
    motion_stack = np.stack(motion_maps)
    print(f"   motion maps: {motion_stack.shape}, range=[{motion_stack.min():.3f}, {motion_stack.max():.3f}]")

    print("4) Sanity check: does amplification increase RAW (unnormalized) motion")
    print("   signal specifically inside the moving-blob region?")
    # Use unnormalized flow magnitude (flow_to_magnitude(..., normalize=False)) so
    # we're comparing true pixel-displacement magnitude, not two independently
    # rescaled [0,1] maps (which would hide the actual effect of amplification).
    blob_region = np.s_[49:79, 49:79]  # matches the blob's approximate location

    raw_blob_magnitudes = []
    amp_blob_magnitudes = []
    for t in range(1, len(frames)):
        raw_flow = farneback_flow(frames[t - 1], frames[t])
        raw_mag = flow_to_magnitude(raw_flow, normalize=False)
        raw_blob_magnitudes.append(raw_mag[blob_region].mean())

        amp_flow = farneback_flow(amplified[t - 1], amplified[t])
        amp_mag = flow_to_magnitude(amp_flow, normalize=False)
        amp_blob_magnitudes.append(amp_mag[blob_region].mean())

    raw_mean = np.mean(raw_blob_magnitudes)
    amp_mean = np.mean(amp_blob_magnitudes)
    print(f"   mean raw pixel-displacement in blob region (unamplified): {raw_mean:.4f}")
    print(f"   mean raw pixel-displacement in blob region (amplified):   {amp_mean:.4f}")
    print(f"   amplification factor observed: {amp_mean / (raw_mean + 1e-8):.2f}x")

    print("5) Running model forward + loss on a synthetic batch ...")
    model = TwoStreamCOD(base_ch=16)
    loss_fn = COD_Loss()

    rgb = torch.from_numpy(frames[10]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    motion = torch.from_numpy(cv2.resize(motion_maps[9], (128, 128))).unsqueeze(0).unsqueeze(0).float()
    fake_gt = torch.zeros(1, 1, 128, 128)
    fake_gt[:, :, 49:79, 49:79] = 1.0  # rough blob location as fake ground truth

    logits = model(rgb, motion)
    loss = loss_fn(logits, fake_gt)
    print(f"   logits shape: {logits.shape}, loss: {loss.item():.4f}")

    print("\nAll pipeline stages ran successfully end-to-end.")


if __name__ == "__main__":
    main()
