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

## Files

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

## Known simplifications

- `motion_amplification.py` implements the **linear** (Laplacian-pyramid)
  variant of Eulerian Video Magnification (Wu et al., 2012), not the more
  robust **phase-based** variant yet (Wadhwa et al., 2013), which requires a
  complex steerable pyramid. The linear variant is simpler and fully
  self-contained; It will be swapped out later if more fidelity is needed — the rest of the pipeline's interface (list of frames in, list of amplified frames out)
  does not need to change.
- `model.py`'s encoders are small custom CNNs, not pretrained ImageNet
  backbones for the first prototype. Still need to replace `AppearanceEncoder` with a pretrained `torchvision.models.resnet18` or a
  PVTv2 backbone for a meaningful accuracy boost — the fusion and decoder
  logic do not need to change.
- `optical_flow.py` defaults to Farneback (classical, CPU-friendly). The RAFT
  wrapper is provided but requires downloading pretrained weights; it will be more accurate on hard/camouflaged
  cases (this is what SLT-Net and EMIP use as their flow source).
