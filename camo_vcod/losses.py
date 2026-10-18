"""
losses.py

Dual loss used for TwoStreamCOD: plain BCE + soft (probabilistic) binary IoU,
equally weighted (alpha = beta = 1.0) as the project default.

  - BCE on logits directly optimizes per-pixel classification.
  - Soft IoU on predicted probabilities optimizes mask overlap, which plain
    BCE does not; "soft" because it uses sigmoid probabilities instead of a
    thresholded mask, so it stays differentiable.

The legacy boundary-weighted BCE variant (weighted_bce_iou_loss) is kept for
reference; COD_Loss defaults to the plain dual formulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_soft_iou_loss(logits: torch.Tensor, target: torch.Tensor,
                      alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """
    logits: (B, 1, H, W) raw model output (pre-sigmoid).
    target: (B, 1, H, W) binary ground-truth mask in {0, 1}.
    Returns scalar: alpha * BCE + beta * (1 - soft_IoU).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)  # mean over batch & pixels

    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = (probs + target - probs * target).sum(dim=(1, 2, 3))
    soft_iou_loss = 1.0 - (inter + 1e-6) / (union + 1e-6)

    return alpha * bce + beta * soft_iou_loss.mean()


class COD_Loss(nn.Module):
    """alpha/beta weight the BCE / soft-IoU terms. Last per-batch component
    values are kept in `last_components` so train.py can log them."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.last_components: dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        union = (probs + target - probs * target).sum(dim=(1, 2, 3))
        soft_iou_loss = 1.0 - (inter + 1e-6) / (union + 1e-6)
        total = self.alpha * bce + self.beta * soft_iou_loss.mean()
        self.last_components = {
            "bce": float(bce.detach()),
            "soft_iou": float(soft_iou_loss.mean().detach()),
        }
        return total


def weighted_bce_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    LEGACY: boundary-weighted BCE + IoU. Not used by COD_Loss anymore; kept so
    older checkpoints/experiments remain reproducible.
    """
    local_avg = F.avg_pool2d(target, kernel_size=31, stride=1, padding=15)
    weight = 1.0 + 5.0 * torch.abs(local_avg - target)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (weight * bce).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = (probs + target - probs * target).sum(dim=(1, 2, 3))
    iou_loss = 1.0 - (inter + 1e-6) / (union + 1e-6)

    return (weighted_bce + iou_loss).mean()


if __name__ == "__main__":
    logits = torch.randn(2, 1, 128, 128)
    target = (torch.rand(2, 1, 128, 128) > 0.7).float()
    loss_fn = COD_Loss()
    print("Loss value:", loss_fn(logits, target).item())
    print("Components:", loss_fn.last_components)
    print("Legacy loss:", weighted_bce_iou_loss(logits, target).item())
