"""
train.py

Training loop for TwoStreamCOD. Assumes preprocess_amplify.py has already been
run so that amplified frames + motion maps exist under --cache_root.

Usage (single dataset):
    python train.py --data_root /path/to/MoCA-Mask/train --cache_root ./cache/train \
                     --epochs 50 --batch_size 8 --lr 1e-4

Usage (multiple datasets merged):
    python train.py \
        --data_roots cad=/kaggle/input/.../CamouflagedAnimalDataset \
                     yuv20k=/kaggle/input/.../YUV20K \
                     moca=/kaggle/working/moca_converted \
        --adapters cad=cad yuv20k=imgs_gt moca=moca_mask \
        --cache_root /kaggle/working/cache --epochs 50 --batch_size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    DatasetSource,
    MoCAMaskDataset,
    MultiSourceVCODDataset,
    cad_adapter,
    make_custom_adapter,
    moca_mask_adapter,
)
from model import TwoStreamCOD
from losses import COD_Loss


ADAPTER_REGISTRY = {
    "moca_mask": moca_mask_adapter,
    "cad": cad_adapter,
    "imgs_gt": make_custom_adapter("Imgs", "GT", frame_ext=".png"),
    "images_masks": make_custom_adapter("images", "masks", frame_ext=".jpg"),
}


def parse_name_pairs(items: list[str] | None) -> dict[str, str]:
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"entries must be name=value, got: {item}")
        name, value = item.split("=", 1)
        result[name] = value
    return result


def build_dataset(args):
    if args.data_roots:
        roots = parse_name_pairs(args.data_roots)
        adapter_names = parse_name_pairs(args.adapters)
        sources = []
        for name, path in roots.items():
            adapter_key = adapter_names.get(name, "moca_mask")
            adapter = ADAPTER_REGISTRY.get(adapter_key)
            if adapter is None:
                raise ValueError(
                    f"Unknown adapter '{adapter_key}' for source '{name}'. "
                    f"Known adapters: {sorted(ADAPTER_REGISTRY)}"
                )
            cache_root = str(Path(args.cache_root) / name)
            sources.append(DatasetSource(name, path, cache_root, adapter))
        return MultiSourceVCODDataset(sources, image_size=args.image_size)

    return MoCAMaskDataset(args.data_root, args.cache_root, image_size=args.image_size)


def main(explicit_args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_roots", type=str, nargs="+", default=None)
    parser.add_argument("--adapters", type=str, nargs="+", default=None)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)

    if explicit_args is not None:
        args = parser.parse_args(explicit_args)
    else:
        args, _ = parser.parse_known_args()

    if not args.data_root and not args.data_roots:
        parser.error("Pass either --data_root (single) or --data_roots (multiple).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(args)
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
