"""
eval.py

Runs a trained model over an evaluation/test split across one or multiple datasets
(MoCA-Mask, CAD, YUV20K, etc.) and reports aggregate and per-source/per-sequence
metrics: MAE, IoU, Dice, S-measure, weighted F-measure.

Supports:
  1. Multi-source with split filtering (matches train.py and Kaggle A/B runs):
     python eval.py --checkpoint kaggle_notebook/reports/runs_amp/best_by_val.pth \\
                    --data_roots cad=../project/Dataset/CamouflagedAnimalDataset \\
                    --adapters cad=cad \\
                    --cache_root ./cache_amp \\
                    --split_dir kaggle_notebook/reports/splits \\
                    --split test \\
                    --image_size 256

  2. Single dataset (backwards compatible):
     python eval.py --checkpoint kaggle_notebook/reports/runs_amp/best_by_val.pth \\
                    --data_root /path/to/data --cache_root ./cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import (
    ADAPTER_REGISTRY,
    DatasetSource,
    MoCAMaskDataset,
    MultiSourceVCODDataset,
)
from infer_demo import load_checkpoint
from metrics import evaluate_all

METRIC_KEYS = ["MAE", "IoU", "Dice", "S_measure", "weighted_F"]


def parse_key_value(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Expected key=value format, got '{item}'")
    k, v = item.split("=", 1)
    return k.strip(), v.strip()


def parse_named_pairs(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    res = {}
    for item in items:
        k, v = parse_key_value(item)
        res[k] = v
    return res


def build_dataset(args, split: str | None = None) -> Dataset:
    if args.data_roots:
        sources = []
        for pair in args.data_roots:
            name, path = parse_key_value(pair)
            adapter_key = args.adapters.get(name, name)
            adapter = ADAPTER_REGISTRY.get(adapter_key)
            if adapter is None:
                raise ValueError(
                    f"Unknown adapter '{adapter_key}' for source '{name}'. "
                    f"Known adapters: {sorted(ADAPTER_REGISTRY)}"
                )
            cache_root = str(Path(args.cache_root) / name) if args.cache_root else ""
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
    """Runs the model over a loader and returns per-frame metric dicts plus (source, sequence) tags."""
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
    out = {k: float(sum(r[k] for r in rows) / n) for k in keys}
    out["n_frames"] = len(rows)
    return out


def group_metrics(per_frame: list[dict], tags: list[tuple[str, str]], by: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row, (source, seq) in zip(per_frame, tags):
        key = source if by == "source" else f"{source}/{seq}"
        groups.setdefault(key, []).append(row)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TwoStreamCOD on a dataset or split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth / .zip)")
    parser.add_argument("--data_root", type=str, default=None, help="Single dataset root (legacy mode)")
    parser.add_argument("--data_roots", type=str, nargs="+", default=None,
                        help="Multiple dataset roots as name=path pairs, e.g. cad=/path/to/cad")
    parser.add_argument("--adapters", type=str, nargs="+", default=None,
                        help="Adapter mapping, e.g. cad=cad moca=moca_mask yuv20k=imgs_gt")
    parser.add_argument("--cache_root", type=str, default=None, help="Directory containing preprocessed motion cache")
    parser.add_argument("--split_dir", type=str, default=None, help="Directory with split json files")
    parser.add_argument("--split", type=str, default="test", help="Split name to evaluate (default: test)")
    parser.add_argument("--image_size", type=int, default=256, help="Image resolution for model input")
    parser.add_argument("--base_ch", type=int, default=32, help="Base channel count for TwoStreamCOD")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--out_report", type=str, default=None, help="Optional output JSON path for test report")
    parser.add_argument("--compare_to", type=str, default=None, help="Optional test_report.json to compare metrics against")
    args = parser.parse_args()

    args.adapters = parse_named_pairs(args.adapters)

    if not args.data_root and not args.data_roots:
        raise ValueError("Must provide either --data_root (single) or --data_roots (multi-source).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device, base_ch=args.base_ch)

    dataset = build_dataset(args, split=args.split if args.split_dir else None)
    print(f"Evaluating on {len(dataset)} samples (split: {args.split if args.split_dir else 'all'}).")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Verify data paths and motion cache.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    per_frame, tags = evaluate_loader(model, loader, device)

    overall = aggregate(per_frame)
    by_source = group_metrics(per_frame, tags, by="source")
    by_sequence = group_metrics(per_frame, tags, by="sequence")

    print("\n" + "=" * 25 + f" EVALUATION RESULTS ({args.split}) " + "=" * 25)
    print(f"Overall (n={overall['n_frames']}):")
    for k in METRIC_KEYS:
        print(f"  {k:<12}: {overall[k]:.4f}")

    if len(by_source) > 1 or list(by_source.keys())[0] != "single":
        print("\nBy Source:")
        for src, metrics in sorted(by_source.items()):
            print(f"  [{src}] (n={metrics['n_frames']}): " +
                  " ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS))

    print("\nBy Sequence:")
    for seq, metrics in sorted(by_sequence.items()):
        print(f"  {seq:<30} IoU={metrics['IoU']:.4f} MAE={metrics['MAE']:.4f} "
              f"Dice={metrics['Dice']:.4f} S={metrics['S_measure']:.4f} Fw={metrics['weighted_F']:.4f} (n={metrics['n_frames']})")

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "overall": overall,
        "by_source": by_source,
        "by_sequence": by_sequence,
    }

    if args.out_report:
        out_p = Path(args.out_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to: {out_p}")

    if args.compare_to:
        ref_p = Path(args.compare_to)
        if ref_p.exists():
            ref = json.loads(ref_p.read_text(encoding="utf-8"))
            print(f"\n" + "=" * 20 + f" COMPARISON WITH REFERENCE ({ref_p.name}) " + "=" * 20)
            print(f"{'Target / Metric':<24} {'Evaluated':<12} {'Reference':<12} {'Diff':<10} {'Status'}")
            print("-" * 68)

            # Compare matching sources
            ref_by_source = ref.get("by_source", {})
            for src, ev_s in by_source.items():
                if src in ref_by_source:
                    ref_s = ref_by_source[src]
                    for m in METRIC_KEYS:
                        if m in ref_s and m in ev_s:
                            ev_v = ev_s[m]
                            ref_v = ref_s[m]
                            diff = ev_v - ref_v
                            status = "MATCH (exact)" if abs(diff) < 1e-4 else ("MATCH (~)" if abs(diff) < 0.03 else "DIFF")
                            print(f"{src + '/' + m:<24} {ev_v:<12.4f} {ref_v:<12.4f} {diff:<+10.4f} {status}")

            # If full dataset was evaluated, compare overall as well
            if len(by_source) == len(ref_by_source):
                ref_overall = ref.get("overall", {})
                for m in METRIC_KEYS:
                    if m in ref_overall and m in overall:
                        ev_v = overall[m]
                        ref_v = ref_overall[m]
                        diff = ev_v - ref_v
                        status = "MATCH (exact)" if abs(diff) < 1e-4 else ("MATCH (~)" if abs(diff) < 0.03 else "DIFF")
                        print(f"{'overall/' + m:<24} {ev_v:<12.4f} {ref_v:<12.4f} {diff:<+10.4f} {status}")


if __name__ == "__main__":
    main()
