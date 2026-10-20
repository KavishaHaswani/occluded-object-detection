"""
eval.py

Runs a trained model over a test dataset (e.g. MoCA-Mask test split, or CAD for
cross-domain testing) and reports mean MAE / IoU / Dice / S-measure / weighted
F-measure - the same metrics reported by SLT-Net, EMIP, and SAM-PM, so results
can be placed directly next to their published numbers.

Usage:
    python eval.py --data_root /path/to/MoCA-Mask/test --cache_root ./cache/test \
                    --checkpoint ./checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MoCAMaskDataset
try:
    from infer_demo import load_checkpoint
except ImportError:
    from camo_vcod.infer_demo import load_checkpoint
from metrics import evaluate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MoCAMaskDataset(args.data_root, args.cache_root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(dataset)} samples.")

    model = load_checkpoint(args.checkpoint, device=device, base_ch=args.base_ch)

    all_metrics = []
    with torch.no_grad():
        for rgb, motion, mask in loader:
            rgb, motion = rgb.to(device), motion.to(device)
            logits = model(rgb, motion)
            probs = torch.sigmoid(logits).cpu().numpy()

            for b in range(probs.shape[0]):
                pred = probs[b, 0]
                gt = mask[b, 0].numpy()
                all_metrics.append(evaluate_all(pred, gt))

    keys = all_metrics[0].keys()
    means = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

    print("\n=== Test-set results ===")
    for k, v in means.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
