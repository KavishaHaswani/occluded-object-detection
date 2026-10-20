# Quantitative & Pipeline Analysis Report: Trained Checkpoints & Evaluation Protocol

## Executive Summary

This report delivers a rigorous analysis of the Eulerian motion-amplified camouflage breaking pipeline across the Kaggle training runs (`runs_amp` and `runs_raw`), the inference runtime (`infer_demo.py`), and the evaluation benchmarks (`eval.py`, `metrics.py`).

Key findings include:
1. **Hyperparameter Consistency**: Model architecture (`base_ch=32`) and amplification parameters ($\alpha=15.0$, $f_{low}=0.2\,\text{Hz}$, $f_{high}=3.0\,\text{Hz}$, $levels=4$, Farneback flow with per-frame max-normalization) are consistent across training scripts.
2. **Inference vs Training Mismatches**: Runtime defaults in `infer_demo.py` differ from training in input resolution (`image_size=352` vs `256`), sliding window dimensions (`window_size=60, overlap=30` vs `40, 20`), and motion map dimensions (`352` vs `256`).
3. **Critical Ground-Truth Anomaly in CAD Chameleon**: In `cad/chameleon`, 100% (42/42) of ground truth masks evaluate to **0 foreground pixels**. The raw PNG labels in CAD chameleon use class indices `{0, 1, 2}` rather than `{0, 255}`. Because `dataset.py` thresholds with `mask > 127`, the ground truth was completely wiped to zero during both training and evaluation.
4. **Do-Nothing Baseline Distortion**: In `metrics.py`, empty-prediction vs empty-ground-truth matches are defined as `IoU = 1.0` and `Dice = 1.0`. Because 52 out of 71 test frames (73.2%) have zero ground-truth pixels, a dummy **all-zero prediction achieves an IoU of 0.7324 and MAE of 0.0080**, deceptively outscoring the trained neural models.
5. **A/B Evaluation on the 71-Frame Test Split**: When evaluated on the 71 local CAD test frames, `runs_amp` achieves `IoU = 0.1890` vs `runs_raw` `0.0702`. However, 42 of these frames are unlabelled chameleon frames (where `runs_amp` scored $3/42$ non-zero IoU solely from 3 completely suppressed predictions) and only 19 frames in `frog` have valid non-empty masks. This sample is statistically insufficient to claim superiority without the full multi-dataset test split.

---

## 1. Settings & Hyperparameter Report

Below is the verified configuration for both `runs_amp` (Run A: motion-amplified) and `runs_raw` (Run B: unamplified motion ablation), cross-referenced across `kaggle_code/`, `camo_vcod/`, root copies, run logs, and Kaggle notebook configurations.

### Configuration Table

| Parameter | `runs_amp` Setting | `runs_raw` Setting | File & Line Reference | Notes & Copy Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **`image_size`** | `256` | `256` | `kaggle_notebook/occluded-object-detection.ipynb:L45, L54` | Passed to `train.py` via `--image_size 256`. Identical in root and `camo_vcod/`. |
| **`base_ch`** | `32` | `32` | `kaggle_notebook/occluded-object-detection.ipynb:L45, L54` | Passed to `train.py` via `--base_ch 32`. Verified via `model.py:L142` (2,135,697 parameters). |
| **`alpha`** | `15.0` | N/A (bypassed) | `kaggle_code/preprocess_amplify.py:L133`, `camo_vcod/preprocess_amplify.py:L133` | Hardcoded in `process_sequence()`. For `runs_raw`, line 126 (`if no_amplify:`) bypasses amplification entirely. |
| **`low_hz`** | `0.2` | N/A (bypassed) | `kaggle_code/preprocess_amplify.py:L133`, `camo_vcod/preprocess_amplify.py:L133` | Hardcoded bandpass low cutoff. Bypassed in `runs_raw`. |
| **`high_hz`** | `3.0` | N/A (bypassed) | `kaggle_code/preprocess_amplify.py:L133`, `camo_vcod/preprocess_amplify.py:L133` | Hardcoded bandpass high cutoff. Bypassed in `runs_raw`. |
| **`pyramid levels`** | `4` | N/A (bypassed) | `kaggle_code/preprocess_amplify.py:L133`, `camo_vcod/preprocess_amplify.py:L133` | Gaussian pyramid decomposition levels. Bypassed in `runs_raw`. |
| **`window_size`** | `40` | `40` | `kaggle_notebook/occluded-object-detection.ipynb:L45, L54` | Passed via CLI `--window_size 40`. Default in script is `60` (`L210`). |
| **`overlap`** | `20` | `20` | `kaggle_notebook/occluded-object-detection.ipynb:L45, L54` | Passed via CLI `--overlap 20`. Default in script is `30` (`L211`). |
| **`fps`** | `24.0` | `24.0` | `kaggle_notebook/occluded-object-detection.ipynb:L45, L54` | Passed via CLI `--fps 24`. Default in script is `24.0` (`L209`). |
| **`flow backend`** | `"farneback"` | `"farneback"` | `kaggle_code/preprocess_amplify.py:L148`, `camo_vcod/preprocess_amplify.py:L148` | Hardcoded call `compute_motion_map(..., backend="farneback")`. |
| **`flow normalization`** | `True` ($[0, 1]$ per-frame) | `True` ($[0, 1]$ per-frame) | `optical_flow.py:L50-L62, L120` | `compute_motion_map()` calls `flow_to_magnitude(flow, normalize=True)`. Divides by `max_val` if $> 1e-6$. |
| **`motion map resize`** | `256` ($\text{float16}$) | `256` ($\text{float16}$) | `kaggle_notebook/occluded-object-detection.ipynb:L45, L54`, `preprocess_amplify.py:L149-L151` | Passed via CLI `--motion_size 256`. Resized using `cv2.resize` and cast to `np.float16`. |
| **Motion Source Domain** | **Amplified frames** | **Unamplified raw frames** | `kaggle_code/preprocess_amplify.py:L126-L134` | In `runs_amp`: `no_amplify=False` $\to$ `amplify_video_in_windows()`. In `runs_raw`: `no_amplify=True` $\to$ `processed = frames`. |

### Differences Between Root, `camo_vcod/`, and `kaggle_code/`
- Prior to updating `dataset.py` and `preprocess_amplify.py`, root versions lacked `--data_roots`, `--motion_size`, and `--no_amplify` ablation flags.
- All three copies of `preprocess_amplify.py`, `dataset.py`, `optical_flow.py`, `motion_amplification.py`, and `model.py` are now **byte-for-byte identical**.

---

## 2. Comparison Against Inference Defaults (`infer_demo.py`)

Evaluating `infer_demo.py` against the settings used to train `runs_amp`:

| Parameter | Training Run (`runs_amp`) | Inference Default (`infer_demo.py`) | Status | Impact / Consequence |
| :--- | :--- | :--- | :--- | :--- |
| **Input Image Size** | `256 x 256` | `352 x 352` (lines 130, 315) | **MISMATCH** | Model receives higher-frequency interpolations than it was optimized on during training. |
| **Motion Map Size** | `256 x 256` (cache) | `352 x 352` (line 219) | **MISMATCH** | Motion map is resized to 352 to match input RGB resolution. |
| **Window Size** | `40` frames | `60` frames (lines 132, 317) | **MISMATCH** | Longer window captures different temporal frequencies; higher memory consumption during window FFT. |
| **Window Overlap** | `20` frames | `30` frames (lines 133, 318) | **MISMATCH** | Maintains 50% overlap ratio, but window spans 2.5 seconds vs 1.67 seconds in training. |
| **Amplification Factor ($\alpha$)** | `15.0` | `15.0` (lines 127, 312) | **MATCH** | Identical Eulerian magnification multiplier. |
| **Bandpass Low Cutoff** | `0.2` Hz | `0.2` Hz (lines 128, 313) | **MATCH** | Identical lower bound. |
| **Bandpass High Cutoff** | `3.0` Hz | `3.0` Hz (lines 129, 314) | **MATCH** | Identical upper bound. |
| **Pyramid Levels** | `4` | `4` (line 171) | **MATCH** | Identical Gaussian pyramid decomposition. |
| **FPS Handling** | Fixed `24.0` | Video metadata `cap.get(CAP_PROP_FPS)` or `24.0` | **ADAPTIVE** | `infer_demo.py` dynamically synchronizes filter cutoff with actual video framerate. |
| **Flow Backend** | `farneback` | `farneback` (line 213) | **MATCH** | Identical optical flow algorithm. |
| **Flow Normalization** | Per-frame max ($[0, 1]$) | Per-frame max ($[0, 1]$) (line 218) | **MATCH** | Identical normalization logic. |
| **Checkpoint Path** | `runs_amp/best_by_val.pth` | `REPO_ROOT / "best_model.zip"` (legacy config) | **MISMATCH** | Backend config pointed to outdated placeholder weights. |

---

## 3. Chameleon Metric Check & Root Cause Analysis

### Frame-by-Frame Results for `cad/chameleon` (`runs_amp/best_by_val.pth`)
Tested on all 42 consecutive evaluation frames:

| Frame Name | GT Foreground Area (px) | Pred Area (>0.5 px) | IoU | Dice | S-measure | Weighted F | Mean Probability | Max Probability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chameleon_006.png` | 0 | 1,232 | 0.0000 | 0.0000 | 0.9714 | 0.9714 | 0.0286 | 0.9154 |
| `chameleon_011.png` | 0 | 917 | 0.0000 | 0.0000 | 0.9738 | 0.9738 | 0.0262 | 0.9175 |
| `chameleon_016.png` | 0 | 191 | 0.0000 | 0.0000 | 0.9884 | 0.9884 | 0.0116 | 0.8343 |
| `chameleon_021.png` | 0 | 1,003 | 0.0000 | 0.0000 | 0.9807 | 0.9807 | 0.0193 | 0.9764 |
| `chameleon_026.png` | 0 | 0 | **1.0000** | **1.0000** | **0.9972** | **0.9972** | 0.0028 | 0.1590 |
| `chameleon_031.png` | 0 | 2,316 | 0.0000 | 0.0000 | 0.9527 | 0.9527 | 0.0473 | 0.9591 |
| `chameleon_036.png` | 0 | 42 | 0.0000 | 0.0000 | 0.9951 | 0.9951 | 0.0049 | 0.7714 |
| `chameleon_041.png` | 0 | 0 | **1.0000** | **1.0000** | **0.9977** | **0.9977** | 0.0023 | 0.2489 |
| `chameleon_046.png` | 0 | 61 | 0.0000 | 0.0000 | 0.9949 | 0.9949 | 0.0051 | 0.9001 |
| `chameleon_051.png` | 0 | 1,708 | 0.0000 | 0.0000 | 0.9643 | 0.9643 | 0.0357 | 0.9899 |
| `chameleon_056.png` | 0 | 339 | 0.0000 | 0.0000 | 0.9892 | 0.9892 | 0.0108 | 0.9258 |
| `chameleon_061.png` | 0 | 1,162 | 0.0000 | 0.0000 | 0.9768 | 0.9768 | 0.0232 | 0.9688 |
| `chameleon_066.png` | 0 | 1,229 | 0.0000 | 0.0000 | 0.9733 | 0.9733 | 0.0267 | 0.9683 |
| `chameleon_071.png` | 0 | 925 | 0.0000 | 0.0000 | 0.9787 | 0.9787 | 0.0213 | 0.9409 |
| `chameleon_076.png` | 0 | 927 | 0.0000 | 0.0000 | 0.9834 | 0.9834 | 0.0166 | 0.9820 |
| `chameleon_081.png` | 0 | 728 | 0.0000 | 0.0000 | 0.9871 | 0.9871 | 0.0129 | 0.9844 |
| `chameleon_086.png` | 0 | 868 | 0.0000 | 0.0000 | 0.9838 | 0.9838 | 0.0162 | 0.9810 |
| `chameleon_091.png` | 0 | 597 | 0.0000 | 0.0000 | 0.9838 | 0.9838 | 0.0162 | 0.9548 |
| `chameleon_096.png` | 0 | 517 | 0.0000 | 0.0000 | 0.9879 | 0.9879 | 0.0121 | 0.9534 |
| `chameleon_101.png` | 0 | 1,257 | 0.0000 | 0.0000 | 0.9772 | 0.9772 | 0.0228 | 0.9660 |
| `chameleon_106.png` | 0 | 30 | 0.0000 | 0.0000 | 0.9962 | 0.9962 | 0.0038 | 0.8245 |
| `chameleon_111.png` | 0 | 103 | 0.0000 | 0.0000 | 0.9889 | 0.9889 | 0.0111 | 0.8697 |
| `chameleon_116.png` | 0 | 1,676 | 0.0000 | 0.0000 | 0.9697 | 0.9697 | 0.0303 | 0.9883 |
| `chameleon_121.png` | 0 | 257 | 0.0000 | 0.0000 | 0.9902 | 0.9902 | 0.0098 | 0.7987 |
| `chameleon_126.png` | 0 | 868 | 0.0000 | 0.0000 | 0.9796 | 0.9796 | 0.0204 | 0.9625 |
| `chameleon_131.png` | 0 | 477 | 0.0000 | 0.0000 | 0.9875 | 0.9875 | 0.0125 | 0.9192 |
| `chameleon_136.png` | 0 | 388 | 0.0000 | 0.0000 | 0.9914 | 0.9914 | 0.0086 | 0.9483 |
| `chameleon_141.png` | 0 | 102 | 0.0000 | 0.0000 | 0.9940 | 0.9940 | 0.0060 | 0.9294 |
| `chameleon_146.png` | 0 | 986 | 0.0000 | 0.0000 | 0.9793 | 0.9793 | 0.0207 | 0.9805 |
| `chameleon_151.png` | 0 | 288 | 0.0000 | 0.0000 | 0.9885 | 0.9885 | 0.0115 | 0.8871 |
| `chameleon_156.png` | 0 | 121 | 0.0000 | 0.0000 | 0.9940 | 0.9940 | 0.0060 | 0.9004 |
| `chameleon_161.png` | 0 | 723 | 0.0000 | 0.0000 | 0.9830 | 0.9830 | 0.0170 | 0.9448 |
| `chameleon_166.png` | 0 | 527 | 0.0000 | 0.0000 | 0.9859 | 0.9859 | 0.0141 | 0.9666 |
| `chameleon_171.png` | 0 | 109 | 0.0000 | 0.0000 | 0.9915 | 0.9915 | 0.0085 | 0.8858 |
| `chameleon_176.png` | 0 | 146 | 0.0000 | 0.0000 | 0.9946 | 0.9946 | 0.0054 | 0.9334 |
| `chameleon_181.png` | 0 | 0 | **1.0000** | **1.0000** | **0.9977** | **0.9977** | 0.0023 | 0.2417 |
| `chameleon_186.png` | 0 | 3,647 | 0.0000 | 0.0000 | 0.9408 | 0.9408 | 0.0592 | 0.9805 |
| `chameleon_191.png` | 0 | 846 | 0.0000 | 0.0000 | 0.9733 | 0.9733 | 0.0267 | 0.9501 |
| `chameleon_196.png` | 0 | 926 | 0.0000 | 0.0000 | 0.9772 | 0.9772 | 0.0228 | 0.9830 |
| `chameleon_201.png` | 0 | 465 | 0.0000 | 0.0000 | 0.9878 | 0.9878 | 0.0122 | 0.8944 |
| `chameleon_206.png` | 0 | 14 | 0.0000 | 0.0000 | 0.9973 | 0.9973 | 0.0027 | 0.7125 |
| `chameleon_211.png` | 0 | 32 | 0.0000 | 0.0000 | 0.9959 | 0.9959 | 0.0041 | 0.8409 |

### Empty Ground-Truth Findings
- **Total Empty GT Frames**: **42 out of 42 frames (100.0%)** have `gt_area == 0`.
- **Root Cause**: The CAD dataset is heterogeneous in mask format:
  - Sequences `chameleon`, `glowwormbeetle`, `scorpion1`, `snail`, and `stickinsect` use class index labels (`{0, 1}` or `{0, 1, 2}`).
  - Sequences `frog`, `scorpion2`, `scorpion3`, and `scorpion4` use standard `{0, 255}` binary masks.
  - In `dataset.py:L258`, masks are binarized with:
    ```python
    mask = (mask > 127).astype(np.float32)
    ```
    This condition zeroes out all pixel values equal to `1` or `2`. As a result, the model was trained against blank masks on chameleon and evaluated against blank masks.

### How `metrics.py` Scores the Empty-Empty Case
When both prediction and ground truth have zero foreground pixels:
1. **IoU & Dice** (`metrics.py:L27-L31`):
   ```python
   inter = np.logical_and(pred_binary, gt).sum()  # 0
   union = np.logical_or(pred_binary, gt).sum()   # 0
   iou = inter / union if union > 0 else 1.0       # -> 1.0
   dice = 2 * inter / (...) if (...) > 0 else 1.0  # -> 1.0
   ```
   Because `union == 0`, it defaults to **`1.0`**.
2. **Weighted F-measure** (`metrics.py:L40-L41`):
   ```python
   if gt.sum() == 0:
       return float(1.0 - pred.mean())
   ```
   When the prediction is completely zero, this returns **`1.0`**. (When `pred.mean()` is $\approx 0.002$, it returns $\approx 0.998$).
3. **S-measure** (`metrics.py:L80-L130`):
   Object-similarity and region SSIM both collapse to background-only uniformity, evaluating to **`1.0`**.

Because exactly 3 frames (`chameleon_026`, `chameleon_041`, `chameleon_181`) had 0 false-positive pixels exceeding threshold $0.5$:
$$\text{IoU}_{\text{chameleon}} = \frac{3 \times 1.0 + 39 \times 0.0}{42} = \frac{3}{42} \approx \mathbf{0.0714}$$
This mathematically explains why the sequence report listed `IoU=0.0714` and `Dice=0.0714`.

---

## 4. Do-Nothing Baseline Comparison

Evaluated on the exact 71 test frames (42 `chameleon` + 29 `frog`):
- **All-Zero Prediction**: Predicts $\hat{y} = 0$ for all pixels.
- **Dataset Mean Mask**: Predicts the spatial mean ground-truth mask ($256 \times 256$, mean pixel value $0.00804$, peak $0.2535$).
- **Model `runs_amp`**: `best_by_val.pth` with motion amplification.
- **Model `runs_raw`**: `best_by_val.pth` without motion amplification.

| Metric | All-Zero Prediction | Dataset Mean Mask | Model `runs_amp` | Model `runs_raw` | Target Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IoU** | **0.7324** | **0.7324** | 0.1890 | 0.0702 | Higher is better |
| **Dice** | **0.7324** | **0.7324** | 0.1941 | 0.0921 | Higher is better |
| **MAE** | **0.0080** | 0.0137 | 0.0179 | 0.0416 | Lower is better |
| **S-measure** | 0.8622 | **0.8799** | 0.8588 | 0.8640 | Higher is better |
| **Weighted F** | **0.7324** | 0.7274 | 0.7230 | 0.7003 | Higher is better |

### Critical Takeaway
- Out of 71 frames in the test set, **52 frames (73.24%) have an empty ground-truth mask** (42 in `chameleon` due to index masking, and 10 in `frog` where the animal has not yet entered the frame).
- An all-zero predictor scores a perfect `1.0` on all 52 empty frames and `0.0` on the 19 non-empty frames:
  $$\text{IoU}_{\text{zero}} = \frac{52}{71} \approx \mathbf{0.7324}$$
- The dummy all-zero predictor achieves a higher IoU, higher Dice, lower MAE, and higher Weighted F than both trained neural models. This demonstrates that standard COD metrics can be heavily inflated on sequences with sparse annotations or empty frames.

---

## 5. A/B Evaluation on the Test Split: `runs_amp` vs `runs_raw`

Evaluated across the 71 test frames using their respective motion caches (`cache_amp` vs `cache_raw`) generated with identical settings:

| Scope | Metric | `runs_amp` (Amplified) | `runs_raw` (Raw Ablation) | Delta ($\Delta$) | Kaggle Reference Match |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Overall (n=71)** | **IoU** | **0.1890** | 0.0702 | **+0.1188** | Match ($\le 0.0006$) |
| | **Dice** | **0.1941** | 0.0921 | **+0.1020** | Match ($\le 0.0008$) |
| | **MAE** | **0.0179** | 0.0416 | **-0.0237** | Match ($\le 0.0001$) |
| | **S-measure** | 0.8588 | **0.8640** | -0.0052 | Match ($\le 0.0003$) |
| | **Weighted F** | **0.7230** | 0.7003 | **+0.0227** | Match ($\le 0.0001$) |
| **`cad/chameleon` (n=42)** | IoU | **0.0714** | 0.0000 | +0.0714 | Match ($\le 0.0001$) |
| | Dice | **0.0714** | 0.0000 | +0.0714 | Match ($\le 0.0001$) |
| | MAE | **0.0162** | 0.0538 | -0.0376 | Match ($\le 0.0001$) |
| | S-measure | **0.9838** | 0.9462 | +0.0376 | Match ($\le 0.0001$) |
| | Weighted F | **0.9838** | 0.9462 | +0.0376 | Match ($\le 0.0001$) |
| **`cad/frog` (n=29)** | IoU | **0.3593** | 0.1720 | **+0.1873** | Match ($\le 0.0002$) |
| | Dice | **0.3718** | 0.2255 | **+0.1463** | Match ($\le 0.0004$) |
| | MAE | **0.0205** | 0.0239 | **-0.0034** | Match ($\le 0.0001$) |
| | S-measure | 0.6777 | **0.7451** | -0.0674 | Match ($\le 0.0005$) |
| | Weighted F | **0.3453** | 0.3443 | +0.0010 | Match ($\le 0.0001$) |

### Statistical Significance Assessment
1. **Sample Size Limitations**:
   The local test set contains only **71 frames across 2 video clips**.
   - In `chameleon` (42 frames), ground truth is 100% empty; the delta reflects only whether 3 frames had zero false positives vs non-zero.
   - In `frog` (29 frames), 10 frames are empty, leaving only **19 frames with genuine camouflaged targets**.
2. **Contrast with Full Kaggle Test Set (911 Frames)**:
   In the full multi-dataset Kaggle test report (`kaggle_notebook/reports/runs_amp/test_report.json` vs `runs_raw/test_report.json`):
   - `runs_amp` overall: `IoU = 0.3366`, `MAE = 0.1113`, `Dice = 0.4381`
   - `runs_raw` overall: `IoU = 0.3523`, `MAE = 0.1072`, `Dice = 0.4594`
   Across 911 frames, `runs_raw` scored +1.57% higher IoU and 0.41% lower MAE.
3. **Conclusion**:
   The apparent advantage of `runs_amp` over `runs_raw` on the local 71 frames (+11.88% IoU) is driven by an unrepresentative 19-frame sample in `frog` and metric edge-cases in `chameleon`. It is **not statistically significant** to conclude that Eulerian amplification universally beats raw optical flow without addressing the mask binarization bug and evaluating across the full multi-clip benchmark.
