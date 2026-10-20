"""
train.py

Training loop for TwoStreamCOD with a proper train/val/test protocol:

  - preprocess_amplify.py must have run first (amplified OR --no_amplify raw
    cache) so motion maps exist under --cache_root.
  - make_split.py writes sequence-level <source>_<split>.json files; passing
    --split_dir makes train/val/test come from those files (no frame leakage -
    whole clips only).
  - The best checkpoint (best_by_val.pth) is selected by VALIDATION mIoU
    (ties broken by lower val MAE), never by training loss.
  - After training, the best checkpoint is loaded ONCE and evaluated on the
    test split, writing <checkpoint_dir>/test_report.json with overall,
    per-source and per-sequence metrics.
  - Everything is seeded (--seed, default 42) for reproducible A/B ablations.

Usage (split training, the normal path):
    python train.py \
        --data_roots moca=/kaggle/working/moca_converted cad=... yuv20k=... \
        --adapters moca=moca_mask cad=cad yuv20k=imgs_gt \
        --cache_root /kaggle/working/cache \
        --split_dir /kaggle/working/splits --split train \
        --checkpoint_dir /kaggle/working/runs/amp \
        --epochs 10 --batch_size 8 --image_size 256

Usage (single dataset, legacy behaviour - train on everything, no val/test):
    python train.py --data_root /path/to/MoCA-Mask/train --cache_root ./cache
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
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
from metrics import evaluate_all
from model import TwoStreamCOD
from losses import COD_Loss


ADAPTER_REGISTRY = {
    "moca_mask": moca_mask_adapter,
    "cad": cad_adapter,
    "imgs_gt": make_custom_adapter("Imgs", "GT", frame_ext=".png"),
    "images_masks": make_custom_adapter("images", "masks", frame_ext=".jpg"),
}

METRIC_KEYS = ["MAE", "IoU", "Dice", "S_measure", "weighted_F"]


def parse_name_pairs(items: list[str] | None) -> dict[str, str]:
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"entries must be name=value, got: {item}")
        name, value = item.split("=", 1)
        result[name] = value
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(args, split: str | None):
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
        return MultiSourceVCODDataset(
            sources, image_size=args.image_size, split_dir=args.split_dir, split=split,
        )

    if args.split_dir:
        raise ValueError(
            "--split_dir is only supported with --data_roots (the single-source "
            "--data_root mode has no named split files)."
        )
    return MoCAMaskDataset(args.data_root, args.cache_root, image_size=args.image_size)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[list[dict], list[tuple[str, str]]]:
    """Runs the model over a loader (shuffle must be False) and returns a list
    of per-frame metric dicts plus matching (source, sequence) tags, taken in
    the same order as dataset.samples."""
    model.eval()
    per_frame: list[dict] = []
    tags: list[tuple[str, str]] = []
    for rgb, motion, mask in loader:
        rgb = rgb.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        logits = model(rgb, motion)
        probs = torch.sigmoid(logits).cpu()
        for i in range(probs.size(0)):
            pred = probs[i, 0].numpy()
            gt = mask[i, 0].numpy()
            per_frame.append(evaluate_all(pred, gt))
            tag = getattr(loader.dataset, "samples", None)
            if tag is not None:
                idx = len(tags)
                s = loader.dataset.samples[idx]
                tags.append((s["source"], s["sequence"]))
            else:
                tags.append(("single", "single"))
    return per_frame, tags


def aggregate(rows: list[dict], keys: list[str] = METRIC_KEYS) -> dict:
    n = max(len(rows), 1)
    out = {k: sum(r[k] for r in rows) / n for k in keys}
    out["n_frames"] = len(rows)
    return out


def group_metrics(per_frame: list[dict], tags: list[tuple[str, str]],
                  by: str) -> dict:
    """by='source' -> {source: {metrics...}}, by='sequence' -> {source/seq: ...}."""
    groups: dict[str, list[dict]] = {}
    for row, (source, seq) in zip(per_frame, tags):
        key = source if by == "source" else f"{source}/{seq}"
        groups.setdefault(key, []).append(row)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def is_better(val: dict, best: dict) -> bool:
    """Best = highest val IoU; ties (within 1e-6) broken by lowest val MAE."""
    if best is None:
        return True
    diou = val["IoU"] - best["IoU"]
    if diou > 1e-6:
        return True
    if abs(diou) <= 1e-6 and val["MAE"] < best["MAE"] - 1e-9:
        return True
    return False


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory with make_split.py outputs (<source>_<split>.json).")
    parser.add_argument("--split", type=str, default=None,
                        help="Which split to train on (train/val/test). Defaults to 'train' "
                             "when --split_dir is given.")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--no_final_test", action="store_true",
                        help="Skip the end-of-training evaluation on the test split.")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Validate every N epochs.")
    parser.add_argument("--arm_name", type=str, default="default",
                        help="Experimental arm identifier (e.g. raw, amp_old, amp_v2).")

    if explicit_args is not None:
        args = parser.parse_args(explicit_args)
    else:
        args, _ = parser.parse_known_args()

    if not args.data_root and not args.data_roots:
        parser.error("Pass either --data_root (single) or --data_roots (multiple).")
    if args.split_dir and not args.split:
        args.split = "train"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {args.seed} | split_dir: {args.split_dir} | arm: {args.arm_name}")

    split_dir_used = args.split_dir if args.data_roots else None
    dataset = build_dataset(args, split=args.split)
    print(f"Training samples: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(
            "No samples found. Did you run preprocess_amplify.py on this "
            "data_root/cache_root pair, and make_split.py for --split_dir?"
        )

    val_dataset = None
    if split_dir_used:
        # Validation drives best-checkpoint selection; the test split is only
        # touched once at the very end (unless --no_final_test).
        val_dataset = build_dataset(args, split=args.val_split)
        print(f"Validation samples: {len(val_dataset)}")
        if len(val_dataset) == 0:
            raise RuntimeError(f"Validation split '{args.val_split}' is empty.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=False, worker_init_fn=seed_worker,
        )

    model = TwoStreamCOD(base_ch=args.base_ch).to(device)
    loss_fn = COD_Loss()  # plain BCE + soft binary IoU, alpha = beta = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "arm_name": args.arm_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "base_ch": args.base_ch,
        "lr": args.lr,
        "window_size": 40,
        "overlap": 20,
        "split": args.split,
        "split_dir": str(args.split_dir) if args.split_dir else None,
        "cache_root": str(args.cache_root),
        "data_roots": args.data_roots,
        "adapters": args.adapters,
    }
    cfg_p = Path(args.cache_root) / "preprocess_config.json"
    if cfg_p.exists():
        try:
            settings["preprocess"] = json.loads(cfg_p.read_text())
        except Exception:
            pass

    best_path = ckpt_dir / "best_by_val.pth"
    last_path = ckpt_dir / "last_model.pth"
    history: list[dict] = []
    best: dict | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_iou = 0.0
        n_seen = 0
        for rgb, motion, mask in loader:
            rgb, motion, mask = rgb.to(device), motion.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(rgb, motion)
            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()

            bs = rgb.size(0)
            running_loss += loss.item() * bs
            running_bce += loss_fn.last_components["bce"] * bs
            running_iou += loss_fn.last_components["soft_iou"] * bs
            n_seen += bs

        scheduler.step()
        epoch_loss = running_loss / max(n_seen, 1)
        rec: dict = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_bce": running_bce / max(n_seen, 1),
            "train_soft_iou": running_iou / max(n_seen, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }

        line = (
            f"Epoch {epoch:03d}/{args.epochs} - loss {epoch_loss:.4f} "
            f"(bce {rec['train_bce']:.4f} + siou {rec['train_soft_iou']:.4f})"
        )

        do_eval = val_loader is not None and (
            epoch % args.eval_every == 0 or epoch == args.epochs
        )
        if do_eval:
            per_frame, _ = evaluate_loader(model, val_loader, device)
            val = aggregate(per_frame)
            rec["val"] = val
            line += (
                f" | val IoU {val['IoU']:.4f} MAE {val['MAE']:.4f} "
                f"S {val['S_measure']:.4f} Fw {val['weighted_F']:.4f}"
            )
            if is_better(val, best):
                best = dict(val)
                best["epoch"] = epoch
                torch.save({
                    "model": model.state_dict(),
                    "settings": settings,
                    "epoch": epoch,
                    "best": best,
                }, best_path)
                line += " *best*"

        print(line)
        history.append(rec)
        payload = {"settings": settings, "epochs": history}
        (ckpt_dir / "history.json").write_text(json.dumps(payload, indent=2))
        torch.save({
            "model": model.state_dict(),
            "settings": settings,
            "epoch": epoch,
            "best": best,
        }, last_path)

    if best is None:
        torch.save({
            "model": model.state_dict(),
            "settings": settings,
            "epoch": args.epochs,
            "note": "no validation; last weights saved as best",
        }, best_path)
        best = {"epoch": args.epochs, "note": "no validation; last weights saved as best"}

    print(f"Training complete. Best by val: {best}")

    if split_dir_used and not args.no_final_test:
        state = torch.load(best_path, map_location=device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict)
        test_dataset = build_dataset(args, split=args.test_split)
        if len(test_dataset) == 0:
            raise RuntimeError(f"Test split '{args.test_split}' is empty.")
        print(f"Final test evaluation on split '{args.test_split}': "
              f"{len(test_dataset)} frames (best checkpoint, epoch {best.get('epoch')})")
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=False, worker_init_fn=seed_worker,
        )
        per_frame, tags = evaluate_loader(model, test_loader, device)
        report = {
            "checkpoint": str(best_path),
            "settings": settings,
            "seed": args.seed,
            "split_dir": str(args.split_dir),
            "test_split": args.test_split,
            "selection": {
                "metric": "val_IoU",
                "tie_break": "val_MAE",
                "best": best,
            },
            "overall": aggregate(per_frame),
            "by_source": group_metrics(per_frame, tags, by="source"),
            "by_sequence": group_metrics(per_frame, tags, by="sequence"),
        }
        report_path = ckpt_dir / "test_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Test overall: {json.dumps(report['overall'])}")
        print(f"Test report written to {report_path}")


if __name__ == "__main__":
    main()
