"""
metrics.py

Standard COD evaluation metrics, matching what SLT-Net / EMIP / SAM-PM report,
so results can be placed directly alongside published numbers:
  - MAE (mean absolute error)
  - IoU / Dice
  - Weighted F-measure (Fbeta^w)
  - S-measure (structure measure, Fan et al. 2017)

All functions take numpy arrays: pred (float, [0,1], predicted probability map)
and gt (float, {0,1}, ground-truth binary mask), same shape (H, W).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage import convolve


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def iou_dice(pred_binary: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred_binary, gt).sum()
    union = np.logical_or(pred_binary, gt).sum()
    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / (pred_binary.sum() + gt.sum()) if (pred_binary.sum() + gt.sum()) > 0 else 1.0
    return float(iou), float(dice)


def weighted_f_measure(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    """
    Weighted F-measure (Margolin et al., 2014) - approximates the standard
    implementation used by COD benchmark toolkits.
    """
    gt = gt.astype(bool)
    if gt.sum() == 0:
        return float(1.0 - pred.mean())

    dst, idx = bwdist(1 - gt, return_indices=True)
    E = np.abs(pred - gt)

    # Weight errors far from the boundary less (pixel-dependent weighting).
    Et = E.copy()
    Et[~gt] = E[idx[0][~gt], idx[1][~gt]]

    K = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    EA = convolve(Et.astype(np.float64), K, mode="nearest")
    MIN_E_EA = np.where(gt & (EA < Et), EA, Et)

    B = np.where(~gt, 2.0 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(dst))
    Ew = MIN_E_EA * B

    TPw = gt.sum() - Ew[gt].sum()
    FPw = Ew[~gt].sum()

    R = 1 - Ew[gt].mean() if gt.sum() > 0 else 0.0
    P = TPw / (TPw + FPw + 1e-8)

    fw = (1 + beta2) * R * P / (R + beta2 * P + 1e-8) if (R + beta2 * P) > 0 else 0.0
    return float(fw)


def s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """
    Structure measure (Fan et al., ICCV 2017). Combines an object-aware term
    (S_o) and a region-aware term (S_r).
    """
    gt = gt.astype(bool)
    y = gt.mean()

    if y == 0:
        return float(1 - pred.mean())
    if y == 1:
        return float(pred.mean())

    s_o = _s_object(pred, gt)
    s_r = _s_region(pred, gt)
    return float(alpha * s_o + (1 - alpha) * s_r)


def _s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    fg = pred * gt
    fg_mean = fg[gt].mean() if gt.sum() > 0 else 0.0
    fg_std = fg[gt].std() if gt.sum() > 0 else 0.0
    o_fg = 2 * fg_mean / (fg_mean**2 + 1 + fg_std + 1e-8)

    bg = (1 - pred) * (~gt)
    bg_mean = bg[~gt].mean() if (~gt).sum() > 0 else 0.0
    bg_std = bg[~gt].std() if (~gt).sum() > 0 else 0.0
    o_bg = 2 * bg_mean / (bg_mean**2 + 1 + bg_std + 1e-8)

    u = gt.mean()
    return u * o_fg + (1 - u) * o_bg


def _centroid(gt: np.ndarray) -> tuple[int, int]:
    if gt.sum() == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    rows = np.any(gt, axis=1)
    cols = np.any(gt, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    cy = int(np.round(gt.nonzero()[0].mean())) if len(r_idx) else gt.shape[0] // 2
    cx = int(np.round(gt.nonzero()[1].mean())) if len(c_idx) else gt.shape[1] // 2
    return cy, cx


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = pred.shape
    n = h * w
    x, y = pred.mean(), gt.mean()
    sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + 1e-8)
    sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + 1e-8)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-8)

    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        return float(alpha / (beta + 1e-8))
    elif alpha == 0 and beta == 0:
        return 1.0
    else:
        return 0.0


def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    cy, cx = _centroid(gt)
    h, w = gt.shape

    def split(arr):
        return arr[:cy, :cx], arr[:cy, cx:], arr[cy:, :cx], arr[cy:, cx:]

    gt_f = gt.astype(np.float64)
    p_lt, p_rt, p_lb, p_rb = split(pred)
    g_lt, g_rt, g_lb, g_rb = split(gt_f)

    area = h * w
    w_lt = (cy * cx) / area
    w_rt = (cy * (w - cx)) / area
    w_lb = ((h - cy) * cx) / area
    w_rb = ((h - cy) * (w - cx)) / area

    parts = [(p_lt, g_lt, w_lt), (p_rt, g_rt, w_rt), (p_lb, g_lb, w_lb), (p_rb, g_rb, w_rb)]
    score = 0.0
    for p_part, g_part, weight in parts:
        if p_part.size == 0:
            continue
        score += weight * _ssim(p_part, g_part)
    return score


def evaluate_all(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    """Runs all metrics and returns a dict, e.g. for logging/aggregation across a test set."""
    pred_binary = (pred > threshold).astype(np.float32)
    iou, dice = iou_dice(pred_binary, gt)
    return {
        "MAE": mae(pred, gt),
        "IoU": iou,
        "Dice": dice,
        "S_measure": s_measure(pred, gt),
        "weighted_F": weighted_f_measure(pred, gt),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = (rng.random((128, 128)) > 0.9).astype(np.float32)
    pred = np.clip(gt + rng.normal(0, 0.1, size=gt.shape), 0, 1).astype(np.float32)
    print(evaluate_all(pred, gt))
