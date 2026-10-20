"""
optical_flow.py

Computes dense optical flow between two consecutive (amplified) frames, and
reduces it to a single-channel motion-magnitude map that is used as the input
to the model's motion stream.

Two backends are provided:
  - farneback_flow(): classical, CPU-only, fast, no training/weights needed
    (cv2.calcOpticalFlowFarneback). Good default for a hackathon / limited GPU.
  - raft_flow(): deep-learning based (Teed & Deng, 2020), much more accurate on
    hard cases but needs a GPU and pretrained weights. Used by prior VCOD work
    (SLT-Net, EMIP) as their motion source. Wrapped so it can be swapped in with
    a single flag if you have the pretrained weights available locally.

IMPORTANT: both frames passed in must be from the SAME domain (both raw, or
both motion-amplified) — never mix a raw frame with an amplified one, since the
amplification itself perturbs pixel values everywhere and would be picked up
as spurious "motion" by the flow estimator.
"""

from __future__ import annotations

import numpy as np
import cv2


def farneback_flow(prev_bgr_uint8: np.ndarray, curr_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    Classical dense optical flow (Farneback, 2003). No GPU / weights required.
    Returns a (H, W, 2) float32 array of per-pixel (dx, dy) displacement.
    """
    prev_gray = cv2.cvtColor(prev_bgr_uint8, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr_uint8, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow  # (H, W, 2)


def flow_to_magnitude(
    flow: np.ndarray,
    normalize: bool = True,
    median_sub: bool = False,
    norm_mode: str = "max",
) -> np.ndarray:
    """
    Converts a (H, W, 2) flow field into a single-channel magnitude map.
    - median_sub: if True, subtracts the median flow vector (u - med(u), v - med(v))
      before computing magnitude, removing camera ego-motion.
    - norm_mode:
        "max": divide by frame's own max (robust to clip-level motion scale differences)
        "percentile": robust per-frame 1-99 percentile scaling to [0, 1]
    """
    u = flow[..., 0]
    v = flow[..., 1]
    if median_sub:
        u = u - np.median(u)
        v = v - np.median(v)
    magnitude = np.sqrt(u ** 2 + v ** 2)

    if normalize:
        if norm_mode == "percentile":
            p1, p99 = np.percentile(magnitude, (1.0, 99.0))
            if p99 > p1 + 1e-6:
                magnitude = np.clip((magnitude - p1) / (p99 - p1), 0.0, 1.0)
            else:
                magnitude = np.zeros_like(magnitude)
        else:
            max_val = magnitude.max()
            if max_val > 1e-6:
                magnitude = magnitude / max_val
    return magnitude.astype(np.float32)


class RaftFlowEstimator:
    """
    Optional deep-learning flow backend (RAFT, Teed & Deng 2020) via torchvision.
    Only instantiate this if torchvision's pretrained RAFT weights are available
    in your environment (they require a one-time download from download.pytorch.org,
    which may not be reachable from a restricted/offline environment — in that
    case, use farneback_flow() above instead, which needs no weights at all).
    """

    def __init__(self, device: str = "cpu", small: bool = True):
        import torch
        from torchvision.models.optical_flow import raft_small, raft_large
        from torchvision.models.optical_flow import Raft_Small_Weights, Raft_Large_Weights

        self.torch = torch
        self.device = device
        if small:
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=True).to(device).eval()
        else:
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=True).to(device).eval()
        self.transforms = weights.transforms()

    def __call__(self, prev_bgr_uint8: np.ndarray, curr_bgr_uint8: np.ndarray) -> np.ndarray:
        torch = self.torch
        prev_rgb = cv2.cvtColor(prev_bgr_uint8, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr_bgr_uint8, cv2.COLOR_BGR2RGB)
        t_prev = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float()
        t_curr = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float()
        t_prev, t_curr = self.transforms(t_prev, t_curr)
        t_prev, t_curr = t_prev.to(self.device), t_curr.to(self.device)
        with torch.no_grad():
            flow_predictions = self.model(t_prev, t_curr)
        flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        return flow


def compute_motion_map(
    prev_frame_uint8: np.ndarray,
    curr_frame_uint8: np.ndarray,
    backend: str = "farneback",
    raft_estimator: "RaftFlowEstimator | None" = None,
    median_sub: bool = False,
    norm_mode: str = "max",
) -> np.ndarray:
    """
    Convenience wrapper: prev/curr should both be AMPLIFIED frames (or both raw in ablation).
    Returns a single-channel, normalized motion-magnitude map (H, W), float32 in [0, 1].
    """
    if backend == "farneback":
        flow = farneback_flow(prev_frame_uint8, curr_frame_uint8)
    elif backend == "raft":
        assert raft_estimator is not None, "Pass a RaftFlowEstimator instance for backend='raft'"
        flow = raft_estimator(prev_frame_uint8, curr_frame_uint8)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return flow_to_magnitude(flow, normalize=True, median_sub=median_sub, norm_mode=norm_mode)
