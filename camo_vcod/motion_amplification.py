"""
motion_amplification.py

Implements Linear Eulerian Video Magnification (Wu et al., 2012, "Eulerian Video
Magnification for Revealing Subtle Changes in the World").

Pipeline per sliding window of frames:
  1. Build a Laplacian pyramid for every frame in the window.
  2. At each pyramid level, take the time series of each pixel across the window
     and apply a temporal bandpass filter (Butterworth) to isolate the frequency
     band where the motion of interest lives (e.g. 0.2-3 Hz for breathing / slow
     limb movement).
  3. Multiply the filtered (bandpassed) signal by an amplification factor alpha.
  4. Add the amplified signal back onto the original Laplacian pyramid and
     collapse the pyramid back into a full-resolution frame.

NOTE ON SIMPLIFICATION: the literature also describes a *phase-based* variant
(Wadhwa et al., 2013) which amplifies motion via phase changes in a complex
steerable pyramid instead of intensity changes in a Laplacian pyramid. Phase-based
magnification is more robust to noise and more physically tied to true pixel
displacement, but requires implementing a complex steerable pyramid (oriented
quadrature filters), which is a much larger undertaking. This module implements
the classical *linear* (Laplacian-pyramid) variant, which is simpler, fully
self-contained, and sufficient to expose small motions well enough for a
downstream optical-flow estimator to pick them up reliably. If more fidelity is
needed later, swap this module out for a phase-based implementation without
changing anything else in the pipeline (the public interface below stays the
same: a list of frames in, a list of amplified frames out).
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.signal import butter, filtfilt


def build_laplacian_pyramid(frame: np.ndarray, levels: int) -> list[np.ndarray]:
    """Builds a Laplacian pyramid for a single frame (float32, HxWxC)."""
    gaussian = frame.copy()
    gauss_pyr = [gaussian]
    for _ in range(levels):
        gaussian = cv2.pyrDown(gaussian)
        gauss_pyr.append(gaussian)

    lap_pyr = []
    for i in range(levels):
        size = (gauss_pyr[i].shape[1], gauss_pyr[i].shape[0])
        upsampled = cv2.pyrUp(gauss_pyr[i + 1], dstsize=size)
        lap = gauss_pyr[i] - upsampled
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1])  # coarsest Gaussian level (residual)
    return lap_pyr


def collapse_laplacian_pyramid(lap_pyr: list[np.ndarray]) -> np.ndarray:
    """Reconstructs a frame from its Laplacian pyramid."""
    frame = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        size = (lap_pyr[i].shape[1], lap_pyr[i].shape[0])
        frame = cv2.pyrUp(frame, dstsize=size) + lap_pyr[i]
    return frame


def temporal_bandpass_filter(
    signal_stack: np.ndarray, fps: float, low_hz: float, high_hz: float, order: int = 3
) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter along the time axis (axis=0).
    signal_stack: (T, H, W, C) array of pixel values across a window of T frames.
    """
    nyquist = fps / 2.0
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.999)
    b, a = butter(order, [low, high], btype="band")
    # filtfilt needs at least ~3*(max(len(a), len(b))) samples; caller ensures window is large enough.
    filtered = filtfilt(b, a, signal_stack, axis=0)
    return filtered


class EulerianMotionAmplifier:
    """
    Amplifies subtle motion in a window of video frames.

    Usage:
        amp = EulerianMotionAmplifier(fps=30, low_hz=0.2, high_hz=3.0, alpha=15, levels=4)
        amplified_frames = amp.amplify_window(list_of_bgr_uint8_frames)
    """

    def __init__(
        self,
        fps: float,
        low_hz: float = 0.2,
        high_hz: float = 3.0,
        alpha: float = 15.0,
        levels: int = 4,
        pyramid_level_for_amplification: int = 2,
    ):
        self.fps = fps
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.alpha = alpha
        self.levels = levels
        # We only amplify a mid spatial-frequency band (not the finest, which is
        # dominated by noise, and not the coarsest, which barely contains texture).
        self.target_level = pyramid_level_for_amplification

    def amplify_window(self, frames_bgr_uint8: list[np.ndarray]) -> list[np.ndarray]:
        """
        frames_bgr_uint8: list of T frames (H, W, 3) uint8, BGR (as read by cv2).
        Returns: list of T amplified frames, same shape/dtype.
        """
        T = len(frames_bgr_uint8)
        min_required = 3 * 8  # rough filtfilt stability margin for order=3 filter
        if T < min_required:
            raise ValueError(
                f"Window too short for stable temporal filtering "
                f"(got {T} frames, need >= {min_required}). Increase window size."
            )

        frames_f = [f.astype(np.float32) / 255.0 for f in frames_bgr_uint8]

        # Build Laplacian pyramids for every frame in the window.
        pyramids = [build_laplacian_pyramid(f, self.levels) for f in frames_f]

        # Stack the target level across time: shape (T, H_l, W_l, C)
        level_stack = np.stack([p[self.target_level] for p in pyramids], axis=0)

        # Temporal bandpass filter + amplify.
        filtered = temporal_bandpass_filter(level_stack, self.fps, self.low_hz, self.high_hz)
        amplified_level = level_stack + self.alpha * filtered

        # Rebuild each frame's pyramid with the amplified target level, then collapse.
        amplified_frames = []
        for t in range(T):
            pyr = list(pyramids[t])
            pyr[self.target_level] = amplified_level[t]
            recon = collapse_laplacian_pyramid(pyr)
            recon = np.clip(recon, 0.0, 1.0)
            amplified_frames.append((recon * 255.0).astype(np.uint8))

        return amplified_frames


def amplify_video_in_windows(
    frames_bgr_uint8: list[np.ndarray],
    fps: float,
    window_size: int = 60,
    overlap: int = 30,
    **amplifier_kwargs,
) -> list[np.ndarray]:
    """
    Processes a full video (list of frames) in overlapping sliding windows, since
    the temporal bandpass filter needs a window of frames to operate on, not a
    single frame. Keeps only the stable "core" of each window to avoid edge
    artifacts from filtfilt, and stitches the windows back into one continuous
    amplified sequence covering every input frame.
    """
    amp = EulerianMotionAmplifier(fps=fps, **amplifier_kwargs)
    n = len(frames_bgr_uint8)
    stride = window_size - overlap
    output = [None] * n

    start = 0
    while start < n:
        end = min(start + window_size, n)
        window = frames_bgr_uint8[start:end]
        if len(window) < 24:  # pad short trailing window by repeating last frame
            pad = [window[-1]] * (24 - len(window))
            window = window + pad
        amplified = amp.amplify_window(window)

        # Keep the central, most stable portion of this window; on the first/last
        # window, keep from the true start/end instead of trimming.
        trim = overlap // 2
        lo = 0 if start == 0 else trim
        hi = len(amplified) if end == n else len(amplified) - trim
        for i in range(lo, min(hi, end - start)):
            global_idx = start + i
            if global_idx < n and output[global_idx] is None:
                output[global_idx] = amplified[i]

        start += stride

    # Safety net: fill any unset frame (shouldn't normally happen) with the original.
    for i in range(n):
        if output[i] is None:
            output[i] = frames_bgr_uint8[i]
    return output
