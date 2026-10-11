"""
train.py

Training loop for TwoStreamCOD. Assumes preprocess_amplify.py has already been
run so that amplified frames + motion maps exist under --cache_root.

Usage:
    python train.py --data_root /path/to/MoCA-Mask/train --cache_root ./cache/train \
                     --epochs 50 --batch_size 8 --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MoCAMaskDataset
from model import TwoStreamCOD
from losses import COD_Loss


def main(explicit_args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    # args = parser.parse_args()
    
    if explicit_args is not None:
        args = parser.parse_args(explicit_args)
    else:
        args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MoCAMaskDataset(args.data_root, args.cache_root, image_size=args.image_size)
    print(f"Training samples: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(
            "No samples found. Did you run preprocess_amplify.py on this data_root/cache_root pair?"
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    model = TwoStreamCOD(base_ch=args.base_ch).to(device)
    loss_fn = COD_Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for rgb, motion, mask in loader:
            rgb, motion, mask = rgb.to(device), motion.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(rgb, motion)
            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch:03d}/{args.epochs} - loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        torch.save(model.state_dict(), ckpt_dir / "last_model.pth")

    print(f"Training complete. Best loss: {best_loss:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
