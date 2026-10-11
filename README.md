# Motion-Amplified Camouflage Breaking

A video camouflaged-object-detection (VCOD) pipeline that fuses appearance
features with a **motion signal computed on Eulerian-amplified frames**, so
objects that are visually indistinguishable from the background (color/texture
matches perfectly) can still be detected via the subtle motion they can't hide.

---

## 📷 Screenshots

<!-- Screenshot Placeholders: drop images into docs/screenshots/ to view -->

| Home View & Upload Flow | Before/After Split Slider |
|:---:|:---:|
| ![Home Page & Upload](docs/screenshots/home_page.png)<br><sub>*Drag-and-drop video upload, motion amplification parameter tuning, and sample gallery.*</sub> | ![Before/After Comparison Slider](docs/screenshots/slider_view.png)<br><sub>*Interactive draggable split slider comparing Appearance-only baseline vs Motion-amplified detection.*</sub> |

| Pipeline Stage Viewer | Architecture & Evaluation Metrics |
|:---:|:---:|
| ![Pipeline Stage Viewer](docs/screenshots/stage_viewer.png)<br><sub>*Synchronized multi-stream stage inspection: Raw Video, Amplified Frames, Flow Heatmap, and Detection Mask.*</sub> | ![Architecture and Metrics](docs/screenshots/architecture_metrics.png)<br><sub>*Two-stream gated fusion diagram and quantitative evaluation metrics table.*</sub> |

---

## 🏗 Pipeline Architecture

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

---

## ⚡ Prerequisites & System Requirements

Before running the application or training pipeline, ensure you have the following installed:

1. **Python 3.10+**: Core backend and deep learning pipeline.
2. **Node.js 18+ & npm**: Frontend web application.
3. **FFmpeg (Required)**: Essential for video preprocessing, MP4 (H.264 / AAC) encoding, cross-browser streaming, and frame extraction.

### Installing FFmpeg

- **macOS** (via Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Windows** (via winget or Chocolatey):
  ```powershell
  winget install Gyan.FFmpeg
  # or
  choco install ffmpeg
  ```

Verify your installation:
```bash
ffmpeg -version
```

---

## 🚀 Quickstart: Single-Command Run

You can set up dependencies and launch both servers simultaneously using a single command:

### 1. Install All Dependencies

```bash
make install
```
*(Alternative: `npm run install:all`)*

This creates a local Python virtual environment (`.venv`), installs `backend/requirements.txt` (PyTorch, OpenCV, FastAPI, Uvicorn, etc.), and runs `npm install` inside `frontend/`.

### 2. Start Both Servers Concurrently

```bash
make dev
```
*(Alternative: `make run` or `npm run dev`)*

This starts:
- **FastAPI Backend**: `http://localhost:8000` (Interactive API docs at `http://localhost:8000/docs`)
- **React Frontend**: `http://localhost:5173`

Both servers stream logs to your terminal and shut down cleanly on `Ctrl+C`.

---

## 🎬 Generating Precomputed Video Samples

The repository includes a batch generation script (`scripts/make_samples.py`) that executes `run_pipeline()` on raw source clips and stores the results in the web app's sample gallery format.

1. Drop any video files (`.mp4`, `.mov`, `.avi`, `.webm`) into `samples_input/`.
2. Run the generator:
   ```bash
   make samples
   # or
   python scripts/make_samples.py --input_dir samples_input --output_dir samples
   ```

For each video, this generates:
```
samples/<sample_name>/
  ├── raw.mp4                # Source video standardized for browser playback
  ├── baseline.mp4           # Model detection without motion signal (appearance only)
  ├── amplified.mp4          # Final motion-amplified detection mask overlay
  ├── amplified_frames.mp4   # Eulerian motion-magnified frames
  ├── flow.mp4               # Farneback optical flow motion-magnitude heatmap
  ├── thumbnail.jpg          # First-frame thumbnail preview
  └── meta.json              # Title, description, and parameter metadata
```

Precomputed samples appear automatically in the "Try a sample" gallery on the Home page and open instantly without reprocessing.

---

## 📁 Repository Structure

| Path | Purpose |
|---|---|
| `Makefile` | Unified CLI targets for development, installation, sample building, and testing. |
| `package.json` | Root npm configuration forwarding development scripts to the Makefile. |
| `motion_amplification.py` | Linear Eulerian Video Magnification (Laplacian pyramid + temporal Butterworth bandpass filter + amplification) across overlapping sliding windows. |
| `optical_flow.py` | Computes dense optical flow (Farneback or RAFT) between consecutive amplified frames, normalized to a single-channel motion map. |
| `model.py` | `TwoStreamCOD`: Appearance encoder + motion encoder + per-scale gated fusion + U-Net decoder. |
| `infer_demo.py` | End-to-end inference engine with video rendering and progress callback integration. |
| `scripts/make_samples.py` | Batch processing script generating sample gallery videos, thumbnails, and `meta.json`. |
| `backend/` | FastAPI service (`app.main:app`) handling video uploads, asynchronous inference, CORS, and HTTP 206 Partial Content video streaming. |
| `frontend/` | React 18 + Vite + TypeScript application with Tailwind CSS, split before/after comparison slider, and synchronized pipeline stage viewer. |
| `losses.py` | Weighted BCE + IoU loss (standard COD loss combination). |
| `metrics.py` | Quantitative metrics (MAE, IoU, Dice, S-measure, weighted F-measure). |
| `dataset.py` | PyTorch `Dataset` for MoCA-Mask layout reading cached amplified frames and motion maps. |
| `preprocess_amplify.py` | Offline dataset preprocessing script to cache amplified frames and flow maps. |
| `train.py` | Model training script. |
| `eval.py` | Model evaluation script reporting aggregate metrics. |
| `test_pipeline.py` | Synthetic smoke test verifying amplification and feature extraction stages. |

---

## 🧪 Synthetic Sanity Check

Run a synthetic smoke test without needing a real dataset:

```bash
make test
# or
python test_pipeline.py
```

This generates a synthetic clip containing a near-invisible drifting camouflage blob, executes Eulerian magnification, computes optical flow, and verifies that the motion signal is measurably boosted (~1.8x boost over raw flow).

---

## 📊 Dataset Training & Evaluation (MoCA-Mask)

Expected sequence directory structure:
```
<data_root>/<seq>/Frame/00000.jpg, 00001.jpg, ...
<data_root>/<seq>/GT/00000.png, 00005.png, ...   (ground-truth masks)
```

1. **Preprocess Dataset** (amplify + compute flow once):
   ```bash
   python preprocess_amplify.py --data_root /path/to/MoCA-Mask/train --cache_root ./cache/train --fps 24
   python preprocess_amplify.py --data_root /path/to/MoCA-Mask/test --cache_root ./cache/test --fps 24
   ```

2. **Train**:
   ```bash
   python train.py --data_root /path/to/MoCA-Mask/train --cache_root ./cache/train --epochs 50 --batch_size 8
   ```

3. **Evaluate**:
   ```bash
   python eval.py --data_root /path/to/MoCA-Mask/test --cache_root ./cache/test --checkpoint ./best_model.zip
   ```

4. **Standalone Demo Video Generation**:
   ```bash
   python infer_demo.py --video your_clip.mp4 --checkpoint ./best_model.zip --output demo_output.mp4
   ```

---

## 🔬 Technical Notes & Design Decisions

- **Linear vs. Phase-Based Magnification**: The current implementation utilizes the linear (Laplacian-pyramid) formulation of Eulerian Video Magnification (Wu et al., 2012) due to its mathematical clarity and zero external dependencies. Phase-based magnification (Wadhwa et al., 2013) can be plugged in without changing the downstream interface.
- **Farneback vs. Deep Flow**: Classical Farneback optical flow is employed by default for real-time CPU friendliness and fast local execution. A RAFT wrapper is provided in `optical_flow.py` for high-precision inference when pretrained weights are loaded.
- **Baseline Definition**: The "Appearance-only" baseline passes identical video frames through the exact same two-stream network with the motion stream zeroed out, isolating the exact performance delta contributed by Eulerian motion magnification.
- **HTTP Range Requests (RFC 7233)**: Backend static endpoints support HTTP 206 Partial Content requests, allowing the browser's video engine to seek, buffer, and synchronize multi-stream playback seamlessly.
