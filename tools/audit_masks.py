#!/usr/bin/env python3
"""
tools/audit_masks.py

Audits ground-truth mask encodings across all sequences in CAD and any other
attached datasets. Reports:
  - Total labeled frames per sequence
  - Unique raw pixel values in ground-truth PNGs
  - Mean foreground fraction under the OLD rule (mask > 127) vs NEW rule (mask > 0 if max <= 2 else mask > 127)
  - Number and percentage of empty masks under both rules
  - Flags sequences with empty masks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_named_roots(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path format, got: {item}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def find_gt_dir(seq_dir: Path) -> Path | None:
    for cand in ["GT", "groundtruth", "masks", "GroundTruth"]:
        p = seq_dir / cand
        if p.is_dir():
            return p
    return None


def audit_sequence(seq_dir: Path) -> dict | None:
    gt_dir = find_gt_dir(seq_dir)
    if not gt_dir:
        return None

    files = sorted([f for f in gt_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                    and not f.name.startswith("._") and f.name != ".DS_Store"])
    if not files:
        return None

    unique_vals = set()
    total_px = 0
    fg_px_old = 0
    fg_px_new = 0
    empty_old = 0
    empty_new = 0

    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3:
            im = im[:, :, 0]
        unique_vals.update(np.unique(im))
        npix = im.size
        total_px += npix

        # Old rule: mask > 127
        m_old = (im > 127).astype(np.float32)
        if m_old.sum() == 0:
            empty_old += 1
        fg_px_old += m_old.sum()

        # New rule: if max <= 2: mask > 0 else mask > 127
        if im.max() <= 2:
            m_new = (im > 0).astype(np.float32)
        else:
            m_new = (im > 127).astype(np.float32)
        if m_new.sum() == 0:
            empty_new += 1
        fg_px_new += m_new.sum()

    n = len(files)
    return {
        "sequence": seq_dir.name,
        "n_frames": n,
        "unique_vals": sorted(list(unique_vals)),
        "empty_old": empty_old,
        "empty_old_pct": (empty_old / n) * 100.0 if n > 0 else 0.0,
        "empty_new": empty_new,
        "empty_new_pct": (empty_new / n) * 100.0 if n > 0 else 0.0,
        "fg_frac_old": fg_px_old / (total_px + 1e-9),
        "fg_frac_new": fg_px_new / (total_px + 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit ground truth mask binarization across datasets.")
    parser.add_argument("--data_roots", nargs="*", default=None,
                        help="Optional dataset roots as name=path pairs, e.g. cad=/path/to/cad")
    parser.add_argument("--mode", choices=["both", "old", "new"], default="both",
                        help="Mode: 'old' (before fix), 'new' (after fix), or 'both' (side-by-side comparison)")
    args = parser.parse_args()

    candidates = {}
    if args.data_roots:
        candidates = parse_named_roots(args.data_roots)
    else:
        # Default local detection
        cad_local = ROOT.parent / "project" / "Dataset" / "CamouflagedAnimalDataset"
        if cad_local.exists():
            candidates["cad"] = str(cad_local)
        cad_in_repo = ROOT / "data" / "cad"
        if cad_in_repo.exists():
            candidates["cad"] = str(cad_in_repo)
        moca_local = ROOT / "data" / "moca"
        if moca_local.exists():
            candidates["moca"] = str(moca_local)

    if not candidates:
        print("No datasets found to audit. Pass --data_roots name=path.")
        return

    print("=" * 100)
    if args.mode == "old":
        print("DATASET MASK AUDIT REPORT: BEFORE FIX (OLD RULE: mask > 127)")
    elif args.mode == "new":
        print("DATASET MASK AUDIT REPORT: AFTER FIX (NEW RULE: mask > 0 if max <= 2 else mask > 127)")
    else:
        print("DATASET MASK AUDIT REPORT: COMPARISON (BEFORE vs AFTER FIX)")
    print("=" * 100)

    for ds_name, ds_path_str in candidates.items():
        ds_path = Path(ds_path_str)
        if not ds_path.is_dir():
            print(f"Skipping {ds_name}: {ds_path} is not a directory.")
            continue

        print(f"\n--- Dataset: {ds_name.upper()} ({ds_path}) ---")
        seq_dirs = sorted([d for d in ds_path.iterdir() if d.is_dir()])
        
        results = []
        for sdir in seq_dirs:
            res = audit_sequence(sdir)
            if res:
                results.append(res)

        if not results:
            print("  No sequences with ground-truth directories found.")
            continue

        if args.mode == "old":
            header = f"{'Sequence':<18} | {'Frames':>6} | {'Unique Raw Vals':<14} | {'Empty Masks':>11} | {'Mean FG %':>10} | {'Status'}"
            print(header)
            print("-" * len(header))
            tot_frames = sum(r['n_frames'] for r in results)
            tot_empty = sum(r['empty_old'] for r in results)
            for r in results:
                flag = "[ALL EMPTY]" if r["empty_old"] == r["n_frames"] else ("[EMPTY FRAMES]" if r["empty_old"] > 0 else "[OK]")
                print(f"{r['sequence']:<18} | {r['n_frames']:>6} | {str(r['unique_vals']):<14} | "
                      f"{r['empty_old']:>4} ({r['empty_old_pct']:>4.1f}%) | "
                      f"{r['fg_frac_old']*100:>9.2f}% | {flag}")
            print("-" * len(header))
            print(f"{'TOTAL':<18} | {tot_frames:>6} | {'-':<14} | "
                  f"{tot_empty:>4} ({(tot_empty/tot_frames)*100:>4.1f}%) |")
        elif args.mode == "new":
            header = f"{'Sequence':<18} | {'Frames':>6} | {'Unique Raw Vals':<14} | {'Empty Masks':>11} | {'Mean FG %':>10} | {'Status'}"
            print(header)
            print("-" * len(header))
            tot_frames = sum(r['n_frames'] for r in results)
            tot_empty = sum(r['empty_new'] for r in results)
            for r in results:
                flag = "[ALL EMPTY]" if r["empty_new"] == r["n_frames"] else (f"[PARTIAL EMPTY: {r['empty_new']}]" if r["empty_new"] > 0 else "[OK]")
                print(f"{r['sequence']:<18} | {r['n_frames']:>6} | {str(r['unique_vals']):<14} | "
                      f"{r['empty_new']:>4} ({r['empty_new_pct']:>4.1f}%) | "
                      f"{r['fg_frac_new']*100:>9.2f}% | {flag}")
            print("-" * len(header))
            print(f"{'TOTAL':<18} | {tot_frames:>6} | {'-':<14} | "
                  f"{tot_empty:>4} ({(tot_empty/tot_frames)*100:>4.1f}%) |")
        else:
            header = f"{'Sequence':<18} | {'Frames':>6} | {'Unique Raw Vals':<14} | {'Empty (Old)':>11} | {'Empty (New)':>11} | {'FG % (Old)':>10} | {'FG % (New)':>10} | {'Flag'}"
            print(header)
            print("-" * len(header))
            total_frames = 0
            total_empty_old = 0
            total_empty_new = 0
            for r in results:
                total_frames += r["n_frames"]
                total_empty_old += r["empty_old"]
                total_empty_new += r["empty_new"]
                flag = ""
                if r["empty_new"] == r["n_frames"]:
                    flag = "[ALL EMPTY]"
                elif r["empty_new"] > 0:
                    flag = f"[PARTIAL EMPTY: {r['empty_new']}]"
                elif r["empty_old"] > 0 and r["empty_new"] == 0:
                    flag = "[FIXED: ZEROED BY OLD RULE]"
                print(f"{r['sequence']:<18} | {r['n_frames']:>6} | {str(r['unique_vals']):<14} | "
                      f"{r['empty_old']:>4} ({r['empty_old_pct']:>4.1f}%) | "
                      f"{r['empty_new']:>4} ({r['empty_new_pct']:>4.1f}%) | "
                      f"{r['fg_frac_old']*100:>9.2f}% | {r['fg_frac_new']*100:>9.2f}% | {flag}")
            print("-" * len(header))
            print(f"{'TOTAL / SUMMARY':<18} | {total_frames:>6} | {'-':<14} | "
                  f"{total_empty_old:>4} ({(total_empty_old/total_frames)*100:>4.1f}%) | "
                  f"{total_empty_new:>4} ({(total_empty_new/total_frames)*100:>4.1f}%) |")


if __name__ == "__main__":
    main()
