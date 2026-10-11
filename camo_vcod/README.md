# Motion-Amplified Camouflage Breaking

A video camouflaged-object-detection (VCOD) pipeline that fuses appearance
features with a **motion signal computed on Eulerian-amplified frames**, so
objects that are visually indistinguishable from the background (color/texture
matches perfectly) can still be detected via the subtle motion they can't hide.

## Pipeline overview

```
raw video frames
       │
       ├──────────────────────────────┐
       │                               │
   (Stream A: appearance)     (Stream B: motion)
       │                               │
       │                    Eulerian motion amplification
       │                    (motion_amplification.py)
       │                               │
       │                    optical flow on AMPLIFIED
       │                    frame pairs (optical_flow.py)
       │                               │
       │                    motion-magnitude map
       │                               │
       └──────────────┬────────────────┘
                       │
              Two-stream network (model.py)
        AppearanceEncoder + MotionEncoder
              + per-scale GatedFusion
              + U-Net decoder
                       │
              segmentation mask
```

## Your specific 4-dataset setup (Kaggle)

Based on inspecting the actual uploaded structures, here's the final status of each:

**CAD (Camouflaged Animal Dataset)** — works as-is:
```python
from dataset import DatasetSource, cad_adapter
DatasetSource("cad",
    "/kaggle/input/datasets/tanushkalokegaonkar/camouflaged-animal-dataset/CamouflagedAnimalDataset",
    "/kaggle/working/cache/cad", cad_adapter)
```
`cad_adapter` handles CAD's mismatched frame/GT naming (`frames/chameleon_001.png` vs.
`groundtruth/001_gt.png`) and filters out the `._`-prefixed macOS metadata files
present throughout this dataset.

**YUV20K** — works as-is:
```python
from dataset import DatasetSource, make_custom_adapter
DatasetSource("yuv20k", "/kaggle/input/datasets/tanushkalokegaonkar/yuv20k/YUV20K",
    "/kaggle/working/cache/yuv20k", make_custom_adapter("Imgs", "GT", frame_ext=".png"))
```

**MoCA** — needs a one-time conversion first, because its ground truth is a single
VIA-exported CSV of bounding boxes (`Annotations/annotations.csv`), not per-frame
mask images:
```bash
python convert_moca_csv_to_masks.py \
    --moca_root /kaggle/input/datasets/ivanomelchenkoim11/moca-dataset/MoCA \
    --output_root /kaggle/working/moca_converted
```
This rasterizes each box into a mask PNG and writes a `Frame`/`GT`-style tree at
`--output_root`, **symlinking** the (large) original JPEGImages rather than
copying them — only the small generated masks are new files on disk. It also
prints a breakdown of each frame's annotated motion type
(`locomotion`/`subtle_motion`/`still`) from the CSV's attribute field — the
`subtle_motion` frames are exactly the hardest, most camouflage-relevant cases,
worth using as a targeted evaluation subset if you want a sharper ablation story.
Once converted, use it like any other dataset:
```python
from dataset import DatasetSource, moca_mask_adapter
DatasetSource("moca", "/kaggle/working/moca_converted",
    "/kaggle/working/cache/moca", moca_mask_adapter)
```
Note: MoCA masks derived this way are **rectangular** (from bounding boxes), not
pixel-precise segmentation masks like the other three datasets — expect IoU/Dice
numbers on this source specifically to run lower even for a correct prediction,
since a tight true segmentation will never perfectly match a rectangle. Consider
reporting MoCA's metrics separately rather than pooled with the others.

**MSVCOD** — not usable for training/evaluation as currently uploaded. This
dataset's Kaggle upload only contains `resources/` (a handful of preview
`.webm` clips) and an `index.html` — this is the project's GitHub Pages preview
material, not the annotated benchmark (no masks present anywhere). Two options:
1. Check for a more complete upload of MSVCOD elsewhere on Kaggle, or the
   dataset's original release location, if you need it for real training/eval numbers.
2. Use these clips purely as extra **qualitative demo footage** — extract frames
   with OpenCV (`cv2.VideoCapture`) and run your *already-trained* model on them
   for the hackathon demo video; just don't expect ground truth to score against.

**Putting the 3 usable sources together:**
```python
from dataset import MultiSourceVCODDataset, DatasetSource, cad_adapter, moca_mask_adapter, make_custom_adapter

sources = [
    DatasetSource("cad", ".../CamouflagedAnimalDataset", "/kaggle/working/cache/cad", cad_adapter),
    DatasetSource("yuv20k", ".../YUV20K", "/kaggle/working/cache/yuv20k", make_custom_adapter("Imgs", "GT", ".png")),
    DatasetSource("moca", "/kaggle/working/moca_converted", "/kaggle/working/cache/moca", moca_mask_adapter),
]
dataset = MultiSourceVCODDataset(sources, image_size=352)
```

## Files (updated)

| File | Purpose |
|---|---|
| `convert_moca_csv_to_masks.py` | One-time conversion: MoCA's VIA-format CSV bounding boxes → mask PNGs, normalized into the standard Frame/GT layout via symlinks. |

| File | Purpose |
|---|---|
| `motion_amplification.py` | Linear Eulerian Video Magnification (Laplacian pyramid + temporal Butterworth bandpass filter + amplification), run in overlapping sliding windows across a full video. |
| `optical_flow.py` | Computes dense optical flow (Farneback by default; optional RAFT wrapper) **between consecutive amplified frames**, reduced to a single-channel motion-magnitude map. |
| `model.py` | `TwoStreamCOD`: appearance encoder + motion encoder + gated fusion + U-Net decoder. |
| `losses.py` | Weighted BCE + IoU loss (standard COD loss combination). |
| `metrics.py` | MAE, IoU, Dice, S-measure, weighted F-measure — matches metrics reported by SLT-Net / EMIP / SAM-PM so results are directly comparable. |
| `dataset.py` | PyTorch `Dataset` for MoCA-Mask-style layout, reading cached amplified frames + motion maps. |
| `preprocess_amplify.py` | Run once per dataset: amplifies every sequence and caches amplified frames + flow-derived motion maps to disk. |
| `train.py` | Training loop. |
| `eval.py` | Runs a trained checkpoint over a test set, reports aggregate metrics. |
| `infer_demo.py` | End-to-end demo: takes a raw video, produces a side-by-side output video (appearance-only baseline vs. motion-amplified pipeline). |
| `test_pipeline.py` | Synthetic smoke test — no real dataset needed, verifies every stage runs and that amplification measurably boosts the motion signal. |

## Using multiple datasets together (e.g. 4 datasets attached on Kaggle)

`dataset.py` supports merging several separately-hosted datasets (different
folders, different naming conventions) **without physically copying them**.
Each dataset stays where it is; only a lightweight index of file paths gets
combined in memory.

1. **Preprocess each dataset**, writing each one's cache to its own named
   subfolder under a shared cache root:
   ```bash
   python preprocess_amplify.py \
       --data_roots moca_mask=/kaggle/input/moca-mask/train \
                    cad=/kaggle/input/cad-dataset \
                    msvcod=/kaggle/input/msvcod/train \
                    yuv20k=/kaggle/input/yuv20k/train \
       --cache_root /kaggle/working/cache --fps 24
   ```
   This auto-detects the frame-folder name per sequence (tries `Frame`,
   `images`, `img`, `frames`, `JPEGImages`, ...) so it copes with datasets that
   don't all use the same convention.

2. **Merge them in code** — build one `DatasetSource` per dataset, using the
   default `moca_mask_adapter` for `Frame`/`GT`-style layouts, or
   `make_custom_adapter(frame_subdir, gt_subdir, frame_ext)` for anything else:
   ```python
   from dataset import MultiSourceVCODDataset, DatasetSource, moca_mask_adapter, make_custom_adapter

   sources = [
       DatasetSource("moca_mask", "/kaggle/input/moca-mask/train",
                     "/kaggle/working/cache/moca_mask", moca_mask_adapter),
       DatasetSource("cad", "/kaggle/input/cad-dataset",
                     "/kaggle/working/cache/cad", moca_mask_adapter),
       DatasetSource("msvcod", "/kaggle/input/msvcod/train",
                     "/kaggle/working/cache/msvcod",
                     make_custom_adapter("images", "masks", frame_ext=".jpg")),
       DatasetSource("yuv20k", "/kaggle/input/yuv20k/train",
                     "/kaggle/working/cache/yuv20k", moca_mask_adapter),
   ]
   dataset = MultiSourceVCODDataset(sources, image_size=352)
   print(dataset.source_breakdown())  # e.g. {'moca_mask': 19313, 'cad': 836, ...}
   ```
   Feed this `dataset` object into `DataLoader` exactly like the single-dataset
   version — `train.py`/`eval.py` don't need to change, since it's the same
   `Dataset` interface either way.

3. **Before you assume it just works** — always run
   `!find /kaggle/input -maxdepth 3 -type d` first to check each dataset's
   actual folder names, since MoCA-Mask/CAD/MSVCOD/etc. don't necessarily use
   identical conventions. If a dataset's frame folder isn't auto-detected, add
   its name to `FRAME_SUBDIR_CANDIDATES` in `preprocess_amplify.py`, and use
   `make_custom_adapter(...)` with its exact folder names in `dataset.py`.

## Setup

```bash
pip install torch torchvision opencv-python-headless numpy scipy
```

## Quick sanity check (no dataset required)

```bash
python test_pipeline.py
```
This generates a synthetic video with a near-invisible drifting blob, runs
amplification + optical flow + the model, and prints an "amplification factor"
showing the measured motion signal is boosted (verified during development to
give ~1.8x higher raw flow magnitude in the moving region vs. unamplified).

## Running on real data (MoCA-Mask)

Expected input layout per sequence:
```
<data_root>/<seq>/Frame/00000.jpg, 00001.jpg, ...
<data_root>/<seq>/GT/00000.png, 00005.png, ...   (ground-truth masks, every 5th frame)
```

1. **Preprocess** (amplify + compute flow, once):
   ```bash
   python preprocess_amplify.py --data_root /path/to/MoCA-Mask/train \
                                 --cache_root ./cache/train --fps 24
   python preprocess_amplify.py --data_root /path/to/MoCA-Mask/test \
                                 --cache_root ./cache/test --fps 24
   ```

2. **Train**:
   ```bash
   python train.py --data_root /path/to/MoCA-Mask/train --cache_root ./cache/train \
                    --epochs 50 --batch_size 8
   ```

3. **Evaluate** (report MAE / IoU / Dice / S-measure / weighted F-measure):
   ```bash
   python eval.py --data_root /path/to/MoCA-Mask/test --cache_root ./cache/test \
                   --checkpoint ./checkpoints/best_model.pth
   ```
   Run the same command against a CAD-formatted test split for cross-domain results.

4. **Demo video** (the hackathon deliverable — side-by-side comparison):
   ```bash
   python infer_demo.py --video your_clip.mp4 --checkpoint ./checkpoints/best_model.pth \
                         --output demo_output.mp4
   ```

## Ablation to run for the hackathon writeup

To isolate whether amplification specifically helps (not just "any" motion
stream), train and evaluate a second model where `optical_flow.py` is fed raw
(non-amplified) frame pairs instead of amplified ones. Compare against:
- **SLT-Net** (implicit motion modeling, no explicit flow)
- **EMIP** (explicit raw-flow + interactive prompting)
- **This pipeline with raw flow** (ablation)
- **This pipeline with amplified flow** (full method)

If the last beats the third on the same test set, that isolates the
amplification step as the source of improvement.

## Known simplifications (documented, not hidden)

- `motion_amplification.py` implements the **linear** (Laplacian-pyramid)
  variant of Eulerian Video Magnification (Wu et al., 2012), not the more
  robust **phase-based** variant (Wadhwa et al., 2013), which requires a
  complex steerable pyramid. The linear variant is simpler and fully
  self-contained; swap it out later if more fidelity is needed — the rest of
  the pipeline's interface (list of frames in, list of amplified frames out)
  does not need to change.
- `model.py`'s encoders are small custom CNNs, not pretrained ImageNet
  backbones, because this was built in an environment without access to
  pretrained-weight download servers. If you have internet access, replace
  `AppearanceEncoder` with a pretrained `torchvision.models.resnet18` or a
  PVTv2 backbone for a meaningful accuracy boost — the fusion and decoder
  logic do not need to change.
- `optical_flow.py` defaults to Farneback (classical, CPU-friendly). The RAFT
  wrapper is provided but requires downloading pretrained weights; use it if
  you have GPU + internet access, since it's more accurate on hard/camouflaged
  cases (this is what SLT-Net and EMIP use as their flow source).
