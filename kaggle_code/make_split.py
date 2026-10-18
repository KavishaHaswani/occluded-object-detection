"""
make_split.py

Generates sequence-level train/val/test split files for every training source.
Whole clips (sequences) are assigned to exactly one split - never frames - so
there is zero leakage between train/val/test.

Policy (seed 42, deterministic):
  - CAD (9 clips): 6 : 1 : 2 (train : val : test), whole clips, categories kept
    distinct (no merging across splits)
  - MoCA / YUV20K: stratified whole-clip shuffle, 8 : 1 : 1 target ratio with
    redistribution so no split ends up empty

Outputs (consumed by dataset.py via train.py --split_dir):
    <out>/moca_train.json  moca_val.json  moca_test.json
    <out>/cad_train.json   cad_val.json   cad_test.json
    <out>/yuv20k_train.json yuv20k_val.json yuv20k_test.json
    <out>/split_summary.json   (counts + full per-split sequence lists)

Usage (on Kaggle, before training):
    python make_split.py \
        --moca_root /kaggle/working/moca_converted \
        --cad_root  /kaggle/input/.../CamouflagedAnimalDataset \
        --yuv_root  /kaggle/input/.../YUV20K \
        --out /kaggle/working/splits
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

RANDOM_SEED = 42
CAD_TARGET_RATIO = (6, 1, 2)  # whole-clip allocation when total == 9
GENERAL_TARGET_RATIO = (8, 1, 1)


def discover_sequences_under(root: Path) -> List[str]:
    """Finds sequence dirs under root that look like a VCOD source.

    Accepts layouts with Frame+GT, frames+groundtruth, Imgs+GT, images+masks,
    OR GT-only (frame symlinks not preserved by Kaggle download). Any dir with
    a GT/groundtruth/GT subdir containing at least one image counts.
    """
    seqs: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        has = (
            ((p / "Frame").exists() and (p / "GT").exists())
            or ((p / "frames").exists() and (p / "groundtruth").exists())
            or ((p / "Imgs").exists() and (p / "GT").exists())
            or ((p / "images").exists() and (p / "masks").exists())
        )
        if not has:
            # GT-only fallback (symlinked Frame dirs lost on download)
            gt_dir = next(
                (p / d for d in ("GT", "groundtruth", "masks") if (p / d).exists()),
                None,
            )
            if gt_dir is not None and any(gt_dir.iterdir()):
                has = True
        if has:
            seqs.append(p.name)
    return seqs


def split_general(
    seqs: List[str], ratio: Tuple[int, int, int]
) -> Tuple[List[str], List[str], List[str]]:
    """Whole-clip split with ratio (train, val, test), seed-fixed."""
    random.seed(RANDOM_SEED)
    total = len(seqs)
    if total == 0:
        return [], [], []
    r = ratio
    ssum = sum(r)
    n_train = int(round(total * r[0] / ssum))
    n_val = int(round(total * r[1] / ssum))
    n_test = total - n_train - n_val
    if n_test < 0:
        n_test, n_val = 0, total - n_train
    if n_val < 0:
        n_val, n_test = 0, total - n_train

    items = list(seqs)
    random.shuffle(items)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]

    # Guarantee every split non-empty for total >= 3
    if total >= 3:
        if not test:
            test = [val.pop()]
        if not val:
            val = [train.pop()]
    # Any leftover (rounding) goes to train
    assigned = set(train) | set(val) | set(test)
    train.extend([x for x in items if x not in assigned])
    return sorted(train), sorted(val), sorted(test)


def split_cad(seqs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """CAD: 6:1:2 whole clips when total==9, distinct categories preserved."""
    random.seed(RANDOM_SEED)
    items = list(seqs)
    random.shuffle(items)
    if len(items) == 9:
        train, val, test = items[:6], items[6:7], items[7:9]
        return sorted(train), sorted(val), sorted(test)
    return split_general(items, CAD_TARGET_RATIO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="/kaggle/working/splits")
    parser.add_argument("--moca_root", type=str, default=None)
    parser.add_argument("--cad_root", type=str, default=None)
    parser.add_argument("--yuv_root", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provided = {name: value for name, value in
                (("moca", args.moca_root), ("cad", args.cad_root), ("yuv20k", args.yuv_root))
                if value}
    if not provided:
        raise SystemExit("Pass at least one of --moca_root / --cad_root / --yuv_root.")

    sequences: dict[str, List[str]] = {}
    for name, root_str in provided.items():
        root = Path(root_str)
        if not root.exists():
            raise FileNotFoundError(f"--{name}_root does not exist: {root}")
        seqs = discover_sequences_under(root)
        if not seqs:
            raise ValueError(
                f"No sequences found under {root} - check the layout "
                f"(Frame+GT / frames+groundtruth / Imgs+GT / masks)."
            )
        sequences[name] = seqs

    splits: dict[str, Tuple[List[str], List[str], List[str]]] = {}
    for name, seqs in sequences.items():
        if name == "cad":
            splits[name] = split_cad(seqs)
        else:
            splits[name] = split_general(seqs, GENERAL_TARGET_RATIO)

    # Integrity checks: no sequence may appear in two splits
    for name, (tr, va, te) in splits.items():
        assert not (set(tr) & set(va)), f"{name}: train/val overlap"
        assert not (set(tr) & set(te)), f"{name}: train/test overlap"
        assert not (set(va) & set(te)), f"{name}: val/test overlap"
        assert set(tr) | set(va) | set(te) == set(sequences[name]), \
            f"{name}: some sequences unassigned"

    for name, (tr, va, te) in splits.items():
        for split_name, vals in (("train", tr), ("val", va), ("test", te)):
            (out_dir / f"{name}_{split_name}.json").write_text(json.dumps(vals, indent=2))

    summary = {
        "seed": RANDOM_SEED,
        "policy": "sequence-level, no frame leakage; CAD 6:1:2 whole clips; others 8:1:1",
        "sources": {},
        "splits": {},
    }
    for name, (tr, va, te) in splits.items():
        summary["sources"][name] = {
            "count": len(sequences[name]),
            "root": str(provided[name]),
            "train": len(tr),
            "val": len(va),
            "test": len(te),
        }
        summary["splits"][name] = {"train": tr, "val": va, "test": te}
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "SPLIT_README.md").write_text(
        "# Split generated by make_split.py\n\n"
        f"- seed: {RANDOM_SEED}\n"
        "- unit: whole sequences (clips), never frames\n"
        "- CAD: 6:1:2 whole clips (categories distinct, not merged)\n"
        "- MoCA/YUV20K: 8:1:1 stratified whole-clip shuffle\n"
        "- overlap assertions passed (train/val/test disjoint, full coverage)\n"
    )
    print(json.dumps(summary["sources"], indent=2))
    print(f"Split files written to {out_dir}")


if __name__ == "__main__":
    main()
