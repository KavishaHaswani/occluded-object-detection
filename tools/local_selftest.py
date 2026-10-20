"""
local_selftest.py

Fast end-to-end smoke test of the train/val/test + raw-cache ablation code,
run entirely on synthetic data (no dataset downloads needed):

  1. builds a tiny 4-sequence MoCA-style dataset (Frame/GT, 24 frames each),
  2. runs make_split.py (seed 42) and checks disjointness + coverage,
  3. runs preprocess_amplify.py twice: amplified cache and --no_amplify raw
     cache, and checks the two caches actually differ,
  4. runs train.py with the split protocol for 2 epochs and checks the
     artifacts: best_by_val.pth, last_model.pth, history.json, test_report.json
     (overall / by_source / by_sequence) plus best-checkpoint selection,
  5. unit-checks split filtering (dataset gets ONLY split sequences),
     split_cad() on a fake 9-clip CAD, and the loss components.

Usage (from repo root):  python tools/local_selftest.py
Everything is created under tools/_selftest/ and can be deleted afterwards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "kaggle_code"
WORK = ROOT / "tools" / "_selftest"
N_FRAMES = 24
GT_STEMS = ["00005", "00010", "00015", "00020"]
SEQS = ["seqA", "seqB", "seqC", "seqD"]


def sh(*args: str) -> str:
    print(f"$ {' '.join(args)}")
    r = subprocess.run([sys.executable, *args], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"FAILED: {args[0]} exited {r.returncode}")
    tail = r.stdout.strip().splitlines()[-8:]
    if tail:
        print("  " + "\n  ".join(tail))
    return r.stdout


def build_data() -> Path:
    data = WORK / "data" / "moca"
    rng = np.random.default_rng(0)
    for seq in SEQS:
        (data / seq / "Frame").mkdir(parents=True, exist_ok=True)
        (data / seq / "GT").mkdir(parents=True, exist_ok=True)
        base = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        for t in range(N_FRAMES):
            frame = np.clip(base.astype(np.int16) + rng.integers(-12, 12, base.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(data / seq / "Frame" / f"{t:05d}.png"), frame)
        for stem in GT_STEMS:
            gt = np.zeros((48, 64), dtype=np.uint8)
            cv2.rectangle(gt, (10, 8), (40, 38), 255, -1)
            cv2.imwrite(str(data / seq / "GT" / f"{stem}.png"), gt)
    return data


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK).mkdir(parents=True)

    # --- unit: CAD 6:1:2 on 9 fake clips + determinism -----------------------
    sys.path.insert(0, str(CODE))
    from make_split import split_cad, split_general, RANDOM_SEED

    fake_cad = [f"clip{i}" for i in range(9)]
    tr, va, te = split_cad(fake_cad)
    assert (len(tr), len(va), len(te)) == (6, 1, 2), (len(tr), len(va), len(te))
    assert set(tr) | set(va) | set(te) == set(fake_cad)
    # Determinism with respect to the RNG (independent of input list order
    # effects beyond what shuffle does) - check first result equals itself.
    a = split_cad(fake_cad)
    assert (tr, va, te) == a, "split_cad not deterministic"

    # --- unit: losses -------------------------------------------------------
    import torch
    from losses import COD_Loss, bce_soft_iou_loss, weighted_bce_iou_loss

    logits = torch.randn(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.7).float()
    loss_fn = COD_Loss()
    total = loss_fn(logits, target)
    comps = loss_fn.last_components
    assert total.ndim == 0
    assert set(comps) == {"bce", "soft_iou"}
    assert abs(total.item() - (comps["bce"] + comps["soft_iou"])) < 1e-5
    assert abs(total.item() - bce_soft_iou_loss(logits, target, 1.0, 1.0).item()) < 1e-5
    legacy = weighted_bce_iou_loss(logits, target)
    assert legacy.ndim == 0
    print("unit: split_cad + losses OK")

    # --- 1) data + split ----------------------------------------------------
    data = build_data()
    splits = WORK / "splits"
    out = sh(str(CODE / "make_split.py"), "--moca_root", str(data), "--out", str(splits))
    print(out)
    summary = json.loads((splits / "split_summary.json").read_text())
    assert summary["seed"] == RANDOM_SEED == 42
    m = summary["splits"]["moca"]
    assert not (set(m["train"]) & set(m["val"])) and not (set(m["train"]) & set(m["test"]))
    assert not (set(m["val"]) & set(m["test"]))
    assert set(m["train"]) | set(m["val"]) | set(m["test"]) == set(SEQS)
    assert (len(m["train"]), len(m["val"]), len(m["test"])) == (2, 1, 1)
    print("make_split OK:", m)

    # --- 2) dataset imports + source construction ---------------------------
    from dataset import DatasetSource, MultiSourceVCODDataset, moca_mask_adapter

    source = DatasetSource("moca", str(data), str(WORK / "cache_amp" / "moca"),
                           moca_mask_adapter)

    # --- 3) preprocess: amplified + raw caches ------------------------------
    common = ["--data_roots", f"moca={data}", "--fps", "8", "--window_size", "24",
              "--overlap", "12", "--motion_size", "32"]
    sh(str(CODE / "preprocess_amplify.py"), *common, "--cache_root", str(WORK / "cache_amp"))
    sh(str(CODE / "preprocess_amplify.py"), *common, "--no_amplify",
       "--cache_root", str(WORK / "cache_raw"))
    amp_files = sorted((WORK / "cache_amp" / "moca" / "seqA" / "motion").glob("*.npy"))
    raw_files = sorted((WORK / "cache_raw" / "moca" / "seqA" / "motion").glob("*.npy"))
    assert len(amp_files) == N_FRAMES - 1 == len(raw_files), (len(amp_files), len(raw_files))
    diffs = 0
    for fa, fr in zip(amp_files, raw_files):
        if not np.allclose(np.load(fa).astype(np.float32), np.load(fr).astype(np.float32)):
            diffs += 1
    assert diffs > 0, "amplified and raw motion caches are identical - ablation broken"
    print(f"preprocess OK: amp vs raw differ in {diffs}/{len(amp_files)} frames of seqA")

    # now the training dataset is buildable through the cache check
    train_ds = MultiSourceVCODDataset([source], image_size=64,
                                      split_dir=str(splits), split="train")
    got = sorted({s["sequence"] for s in train_ds.samples})
    assert got == sorted(m["train"]), (got, m["train"])
    print(f"split filtering OK (train sequences: {got})")

    # --- 4) train.py end-to-end --------------------------------------------
    ckpt = WORK / "runs"
    sh(str(CODE / "train.py"),
       "--data_roots", f"moca={data}", "--adapters", "moca=moca_mask",
       "--cache_root", str(WORK / "cache_amp"),
       "--split_dir", str(splits), "--split", "train",
       "--checkpoint_dir", str(ckpt),
       "--epochs", "2", "--batch_size", "2", "--image_size", "64",
       "--base_ch", "8", "--num_workers", "0")

    for f in ("best_by_val.pth", "last_model.pth", "history.json", "test_report.json"):
        assert (ckpt / f).exists(), f"missing {f}"
    history = json.loads((ckpt / "history.json").read_text())
    assert len(history) == 2 and all("val" in h for h in history), history
    report = json.loads((ckpt / "test_report.json").read_text())
    assert report["selection"]["metric"] == "val_IoU"
    assert report["selection"]["best"]["epoch"] in (1, 2)
    assert report["overall"]["n_frames"] == len(GT_STEMS)
    assert set(report["by_source"]) == {"moca"}
    expected_test_keys = {f"moca/{s}" for s in m["test"]}
    assert set(report["by_sequence"]) == expected_test_keys, report["by_sequence"].keys()
    for k in ("MAE", "IoU", "Dice", "S_measure", "weighted_F"):
        assert 0.0 <= report["overall"][k] <= 1.0, (k, report["overall"][k])
    print("train.py OK:", json.dumps(report["overall"]))

    # --- 5) train.py on the RAW cache (run B path) --------------------------
    ckpt_raw = WORK / "runs_raw"
    sh(str(CODE / "train.py"),
       "--data_roots", f"moca={data}", "--adapters", "moca=moca_mask",
       "--cache_root", str(WORK / "cache_raw"),
       "--split_dir", str(splits), "--split", "train",
       "--checkpoint_dir", str(ckpt_raw),
       "--epochs", "1", "--batch_size", "2", "--image_size", "64",
       "--base_ch", "8", "--num_workers", "0")
    assert (ckpt_raw / "test_report.json").exists()
    print("raw-cache train.py OK")

    print("\nLOCAL SELFTEST PASSED")


if __name__ == "__main__":
    main()
