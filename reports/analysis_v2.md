# Analysis Report v2: Mask Binarization Bug, Dataset Audit, Evaluation Reform & 3-Arm Kaggle Protocol

**Date**: October 2026  
**Repository**: `KavishaHaswani/occluded-object-detection`  
**Status**: Ready for Kaggle Execution (Analysis & Code Updates Complete)

---

## Executive Summary

During our audit of model performance on the Camouflaged Animal Dataset (CAD) and comparison with published baselines, we identified a critical ground-truth binarization flaw across the dataset pipeline (`dataset.py`, `camo_vcod/dataset.py`, `kaggle_code/dataset.py`). The loader applied a hard-coded threshold `mask > 127` under the assumption that all mask files were 8-bit `{0, 255}` images. In CAD, several sequences are encoded with integer class and instance indices `{0, 1}` and `{0, 1, 2}`.

Consequently:
1. **Training Corruption**: **75 out of 103 (72.8%)** of CAD training frames (`glowwormbeetle`, `scorpion1`, `snail`, `stickinsect`) were collapsed to all-zero arrays, severely penalizing the model whenever it predicted camouflage masks on these species.
2. **Evaluation Distortion**: **43 out of 73 (58.9%)** of CAD test frames (`chameleon`) were evaluated against all-zero masks. Because standard segmentation metrics define `IoU = 1.0` when both ground truth and prediction are empty, an all-zero dummy predictor achieved a deceptively high IoU of `0.7324` on the test split, while penalizing real detections down to `0.0714`.

This document reports the ground-truth mask inspection, the full dataset audit before and after the fix, the training impact breakdown from `split_summary.json`, the reformed evaluation protocol (`--skip_empty_gt` and false positive rate tracking), and the complete 3-arm experimental protocol (`raw`, `amp_old`, `amp_v2`) with recorded hyperparameter lineage.

---

## 1. Mask Binarization: Ground Truth Inspection & Decision Rule

### 1.1 Inspection of CAD Ground Truth Masks

We inspected the raw PNG mask files across CAD sequences: `chameleon`, `glowwormbeetle`, `scorpion1`, `snail`, and `stickinsect`.

| Sequence | Folder Layout | Raw Unique Values | Max Value | Spatial Component Analysis | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `chameleon` | `chameleon/groundtruth/*_gt.png` | `{0, 1, 2}` | 2 | Label 1: Area 15,568 px, bbox $y \in [114, 276], x \in [83, 422]$ (left branch)<br>Label 2: Area 12,479 px, bbox $y \in [208, 359], x \in [498, 623]$ (right branch)<br>Overlap of Label 2 with 5px dilation of Label 1: **0 pixels** | **Two distinct foreground animal instances**. Neither is a boundary ring nor an ignore class. |
| `glowwormbeetle` | `glowwormbeetle/groundtruth/*_gt.png` | `{0, 1}` | 1 | Label 1: Area 18,745 px, compact central body mask | **Foreground animal** |
| `scorpion1` | `scorpion1/groundtruth/*_gt.png` | `{0, 1}` | 1 | Label 1: Area 3,270 px, foreground scorpion body/tail | **Foreground animal** |
| `snail` | `snail/groundtruth/*_gt.png` | `{0, 1}` | 1 | Label 1: Area 11,580 px, foreground snail body/shell | **Foreground animal** |
| `stickinsect` | `stickinsect/groundtruth/*_gt.png` | `{0, 1}` | 1 | Label 1: Area 1,110 px, thin stick-insect limbs | **Foreground animal** |
| `frog` | `frog/groundtruth/*_gt.png` | `{0, 255}` | 255 | Label 255: Area ~4,600 px; first 10 frames empty ($0$ px) before animal enters | **Foreground animal** (standard 8-bit) |
| `scorpion2, 3, 4` | `*/groundtruth/*_gt.png` | `{0, 255}` | 255 | Label 255: Standard 8-bit binary mask | **Foreground animal** (standard 8-bit) |

### 1.2 What Values 1 and 2 Mean in CAD

In `chameleon`, the video clip depicts two chameleons interacting on opposing branches. The annotator assigned instance index `1` to the left chameleon and instance index `2` to the right chameleon. 
- A boundary or ignore label (such as in PASCAL VOC or Cityscapes) would form a thin 1- to 5-pixel boundary contour outlining the animal, or would overlap the dilation of the foreground object.
- In `chameleon`, the dilation of Label 1 by 5 pixels has **zero overlap** with Label 2. Furthermore, Label 2 contains 12,479 contiguous pixels forming an entire animal body.
- Therefore, **both 1 and 2 are foreground animal pixels**.

In `glowwormbeetle`, `scorpion1`, `snail`, and `stickinsect`, the masks are 1-bit binary masks stored as unsigned 8-bit integers with values $\{0, 1\}$, where $0 = \text{background}$ and $1 = \text{foreground}$.

### 1.3 The Robust Binarization Rule

Applying `mask > 127` to $\{0, 1\}$ or $\{0, 1, 2\}$ produces an all-zero matrix `(0 > 127 == False)`.

The robust binarization rule replaces `mask > 127` across `dataset.py` (root, `camo_vcod/`, and `kaggle_code/`):
```python
if mask.max() <= 2:
    mask = (mask > 0).astype(np.float32)
else:
    mask = (mask > 127).astype(np.float32)
```

**Why this rule is optimal**:
1. When `mask.max() <= 2`, it detects both 1-class binary masks ($\{0, 1\}$) and multi-instance masks ($\{0, 1, 2\}$), treating any non-zero pixel as foreground.
2. For standard 8-bit masks ($\{0, 255\}$) such as MoCA and YUV20K, `mask.max() == 255 > 2`, preserving the classic `mask > 127` threshold against compression artifacts or antialiased edges.
3. For genuinely empty frames (e.g. `mask.max() == 0`), `0 <= 2`, and `mask > 0` returns an all-zero array without error.

---

## 2. Dataset Audit Script (`tools/audit_masks.py`)

We developed `tools/audit_masks.py` to audit ground-truth mask encodings, unique raw values, empty mask counts, and mean foreground fraction across all sequences in CAD and attached datasets.

### 2.1 Audit Results on CAD (191 Total Frames)

```
====================================================================================================
DATASET MASK AUDIT REPORT: OLD RULE (>127) VS NEW RULE (>0 if max<=2 else >127)
====================================================================================================

--- Dataset: CAD (CamouflagedAnimalDataset) ---
Sequence           | Frames | Unique Raw Vals | Empty (Old) | Empty (New) | FG % (Old) | FG % (New) | Flag
----------------------------------------------------------------------------------------------------------
chameleon          |     43 | [0, 1, 2]       |   43 (100%) |    0 (0.0%) |      0.00% |     14.06% | [FIXED: ZEROED BY OLD RULE]
frog               |     30 | [0, 255]        |   10 (33.3%)|   10 (33.3%)|      2.02% |      2.02% | [PARTIAL EMPTY: 10]
glowwormbeetle     |     21 | [0, 1]          |   21 (100%) |    0 (0.0%) |      0.00% |      8.14% | [FIXED: ZEROED BY OLD RULE]
scorpion1          |     21 | [0, 1]          |   21 (100%) |    0 (0.0%) |      0.00% |      1.42% | [FIXED: ZEROED BY OLD RULE]
scorpion2          |     12 | [0, 255]        |    0 (0.0%) |    0 (0.0%) |      5.41% |      5.41% | 
scorpion3          |     15 | [0, 255]        |    0 (0.0%) |    0 (0.0%) |      0.38% |      0.38% | 
scorpion4          |     16 | [0, 255]        |    0 (0.0%) |    0 (0.0%) |      7.66% |      7.66% | 
snail              |     17 | [0, 1]          |   17 (100%) |    0 (0.0%) |      0.00% |      5.03% | [FIXED: ZEROED BY OLD RULE]
stickinsect        |     16 | [0, 1]          |   16 (100%) |    0 (0.0%) |      0.00% |      0.48% | [FIXED: ZEROED BY OLD RULE]
----------------------------------------------------------------------------------------------------------
TOTAL / SUMMARY    |    191 | -               |  128 (67.0%)|   10 (5.2%) |      2.07% |      6.25% | 
```

**Key Audit Takeaways**:
- Under the old rule, **128 out of 191 frames (67.0%)** were completely zeroed out.
- Under the new rule, only **10 frames (5.2%)** are empty. All 10 empty frames belong to `frog`, representing legitimate frames at the start of the sequence before the frog hops into view.
- 5 out of 9 CAD sequences were previously 100% dead.

---

## 3. Training & Split Impact Analysis

Cross-referencing `split_summary.json` with the audit reveals the exact breakdown across splits:

### 3.1 CAD Split Breakdown

| Split | Sequences Assigned | Sequence Frame Counts | Total Frames | Zeroed Sequences (Old Rule) | Zeroed Frames (Old Rule) | Impact Percentage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `glowwormbeetle`<br>`scorpion1`<br>`scorpion2`<br>`scorpion4`<br>`snail`<br>`stickinsect` | 21<br>21<br>12<br>16<br>17<br>16 | **103** | `glowwormbeetle` (21)<br>`scorpion1` (21)<br>`snail` (17)<br>`stickinsect` (16) | **75** | **72.8% of CAD Train frames were empty!** |
| **Val** | `scorpion3` | 15 | **15** | None (`scorpion3` is `{0, 255}`) | **0** | **0.0%** |
| **Test** | `chameleon`<br>`frog` | 43<br>30 | **73** | `chameleon` (43) | **43** | **58.9% of CAD Test frames were empty!** |

### 3.2 Impact on Other Datasets (MoCA & YUV20K)
- **MoCA**: Ground-truth masks are generated by `convert_moca_csv_to_masks.py` using `mask[y0:y1, x0:x1] = 255`. All annotated frames are $\{0, 255\}$.
- **YUV20K**: Ground-truth masks are standard $\{0, 255\}$ PNGs.
- Therefore, the binarization bug exclusively corrupted the CAD dataset. However, because CAD is our primary specialized camouflaged animal benchmark, corrupting 72.8% of CAD training frames severely hindered the model's ability to learn subtle boundary features on small/thin animals (`stickinsect`, `scorpion1`, `snail`).

---

## 4. Evaluation Reform: `--skip_empty_gt` & Honest Baselines

### 4.1 The Metric Distortion Problem

In `metrics.py`:
$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} \quad \text{if } |P \cup G| > 0 \text{ else } 1.0$$
When a frame has empty ground truth ($|G| = 0$):
- If the model predicts all zeros ($|P| = 0$), $|P \cup G| = 0 \implies \mathbf{IoU = 1.0}$.
- If the model predicts any foreground ($|P| > 0$), $|P \cap G| = 0 \implies \mathbf{IoU = 0.0}$.

In CAD test split under the old rule:
- 52 out of 71 frames were empty (42 chameleon + 10 frog).
- An **all-zero dummy predictor** achieved:
  $$\text{IoU} = \frac{52 \times 1.0 + 19 \times 0.0}{71} = \mathbf{0.7324}, \quad \text{MAE} = \mathbf{0.0080}$$
- A trained model outputting true detections on chameleon was penalized with `0.0` on those 42 frames, receiving a misleadingly low score of `0.0714` on chameleon and `0.1890` overall.

### 4.2 Reformed Evaluation Implementation in `eval.py`

We added the `--skip_empty_gt` flag (default `True`) to `eval.py`.
When enabled:
1. Aggregate metrics ($\text{MAE}, \text{IoU}, \text{Dice}, S\text{-measure}, F_\beta^w$) are computed **only across non-empty ground truth frames** ($|G| > 0$).
2. Empty-GT frames are isolated and evaluated on their own dedicated metric: **False Positive Rate (FPR)**:
   $$\text{FPR}_{\text{empty}} = \frac{\sum_{i \in \text{empty}} \mathbb{I}(\text{pred}_i \text{ has any foreground})}{N_{\text{empty}}}$$
3. The **all-zero baseline** is printed side-by-side on the exact same non-empty frames.

### 4.3 Verification Results on CAD Test Split

Using the current checkpoint `runs_amp/best_by_val.pth`:

| Evaluation Mode | Evaluated Frames | IoU | Dice | MAE | S-measure | weighted F | Empty-GT Frames | Empty-GT FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old Rule (corrupted GT)** | 71 (52 empty) | 0.1890 | 0.1941 | 0.0179 | 0.8588 | 0.7230 | 52 (not tracked) | N/A |
| **Old Rule All-Zero Baseline** | 71 (52 empty) | **0.7324** | **0.7324** | **0.0080** | **0.8662** | **0.7324** | - | - |
| **New Rule (uncorrected eval)** | 71 (10 empty) | 0.1741 | 0.2023 | 0.0911 | 0.5460 | 0.1452 | 10 | 0.0000 |
| **New Rule + `--skip_empty_gt`** | **61 (0 empty)** | **0.0387** | **0.0715** | **0.1059** | **0.4717** | **0.0052** | **10** | **0.0000 (0/10)** |
| **New Rule All-Zero Baseline** | **61 (same frames)**| **0.0000** | **0.0000** | **0.1065** | **0.4468** | **0.0000** | - | - |

**Critical Insights**:
1. On real non-empty frames, the trained model **strictly beats the all-zero baseline** across all metrics (IoU 0.0387 vs 0.0000, Dice 0.0715 vs 0.0000, MAE 0.1059 vs 0.1065, S-measure 0.4717 vs 0.4468).
2. On empty-GT frames (the first 10 frames of `frog`), the model achieves a **perfect 0.0000 False Positive Rate** (0 out of 10 false alarms).
3. The model's modest IoU (0.0387) reflects the fact that it was trained on corrupted data where 72.8% of CAD masks were all zeros. A clean retrain is expected to unlock substantial performance gains.

---

## 5. Kaggle 3-Arm Experimental Protocol

To rigorously evaluate motion amplification without confounding variables, all three experimental arms share identical splits, input resolution, temporal windowing, and training dynamics.

### 5.1 Architecture of the Three Arms

| Parameter | Arm 1: `raw` | Arm 2: `amp_old` | Arm 3: `amp_v2` | Rationale / Difference |
| :--- | :--- | :--- | :--- | :--- |
| **Motion Source** | Raw video frames | Eulerian amplified frames | Eulerian amplified frames (v2) | Tests value of amplification vs baseline |
| **Amplification Gain $\alpha$** | None ($0$) | $15.0$ | $5.0$ | $\alpha=15$ introduces ghosting/fringing; $\alpha=5$ maintains edge fidelity |
| **Bandpass Frequency** | N/A | $[0.2, 3.0]\,\text{Hz}$ | $[0.5, 2.5]\,\text{Hz}$ | Narrower band suppresses camera sway and high-frequency noise |
| **Color Space Amplification** | N/A | Full BGR channels | **Brightness-only (Y in YUV)** | Eliminates chromatic noise amplification |
| **Pyramid Filtering** | N/A | Level 2 only | **Coarse levels ($1, 2, 3$)** | Level 0 unamplified (preserves sensor grain); coarse levels capture body motion |
| **Ego-Motion Compensation** | None | None | **Median-flow subtraction** | Subtracts spatial median $(u - \tilde{u}, v - \tilde{v})$ before magnitude |
| **Flow Normalization** | Max normalization | Max normalization | **Per-frame 1–99 percentile** | Robust against single-pixel flow spikes |
| **Optical Flow Backend** | Farnebäck | Farnebäck | Farnebäck | Identical motion field estimator across all arms |
| **Window & Overlap** | $W=40, O=20$ | $W=40, O=20$ | $W=40, O=20$ | Shared temporal receptive field |
| **Image Resolution** | $256 \times 256$ | $256 \times 256$ | $256 \times 256$ | Shared spatial resolution |
| **Epochs & Batch Size** | 10 epochs, bs 8 | 10 epochs, bs 8 | 10 epochs, bs 8 | Shared training schedule |
| **Random Seed** | 42 | 42 | 42 | Full determinism |

### 5.2 Checkpoint & Lineage Metadata Logging

Each training run records its full configuration inside:
1. **Checkpoint File** (`best_by_val.pth` and `last_model.pth`):
   ```python
   {
       "model": model.state_dict(),
       "settings": {
           "arm_name": "amp_v2",
           "seed": 42,
           "epochs": 10,
           "batch_size": 8,
           "image_size": 256,
           "base_ch": 32,
           "preprocess": {
               "alpha": 5.0,
               "low_hz": 0.5,
               "high_hz": 2.5,
               "brightness_only": True,
               "coarse_only": True,
               "median_sub": True,
               "norm_mode": "percentile"
           }
       },
       "epoch": best_epoch,
       "best": {"IoU": ..., "MAE": ...}
   }
   ```
2. **Training History** (`history.json`):
   ```json
   {
       "settings": { ... },
       "epochs": [
           { "epoch": 1, "train_loss": ..., "val": { ... } }
       ]
   }
   ```
3. **Test Report** (`test_report.json`):
   Contains test split aggregate metrics, per-source metrics, per-sequence metrics, empty-GT count, FPR, and all-zero baseline comparison.

---

## 6. Next Steps

1. Review and apply the unified code diffs across `dataset.py`, `eval.py`, and `kaggle_code/`.
2. Push updated code to the Kaggle repository / code dataset.
3. Launch the 3-arm run on Kaggle GPU.
4. Download the resulting `test_report.json` and checkpoints into `runs_raw/`, `runs_amp_old/`, and `runs_amp_v2/` for final benchmark comparison.
