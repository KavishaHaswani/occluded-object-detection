"""
losses.py

Weighted BCE + IoU loss - the standard loss combination used across COD papers
(SINet, PraNet, ZoomNeXt, EMIP). Pixels near the object boundary are weighted
more heavily in the BCE term (via a local-contrast weight map derived from the
ground-truth mask), and the IoU loss directly optimizes mask overlap, which
plain BCE does not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, 1, H, W) raw model output (pre-sigmoid).
    target: (B, 1, H, W) binary ground-truth mask in {0, 1}.
    """
    # Boundary-aware weight: pixels whose local neighborhood average differs a
    # lot from the pixel itself (i.e. near an edge in the GT mask) get upweighted.
    local_avg = F.avg_pool2d(target, kernel_size=31, stride=1, padding=15)
    weight = 1.0 + 5.0 * torch.abs(local_avg - target)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (weight * bce).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3))

    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = (probs + target - probs * target).sum(dim=(1, 2, 3))
    iou_loss = 1.0 - (inter + 1e-6) / (union + 1e-6)

    return (weighted_bce + iou_loss).mean()


class COD_Loss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return weighted_bce_iou_loss(logits, target)


if __name__ == "__main__":
    logits = torch.randn(2, 1, 128, 128)
    target = (torch.rand(2, 1, 128, 128) > 0.7).float()
    loss_fn = COD_Loss()
    print("Loss value:", loss_fn(logits, target).item())
