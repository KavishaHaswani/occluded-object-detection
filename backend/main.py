"""
FastAPI backend service for motion-amplified camouflage breaking.
Exposes pipeline jobs, live progress monitoring, precomputed sample delivery,
and video streaming with HTTP Range support.
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
import uuid
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Bootstrap root path to import pipeline modules
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_demo import run_pipeline

# Directory paths
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
SAMPLES_DIR = REPO_ROOT / "samples"
CHECKPOINT_PATH = REPO_ROOT / "best_model.zip"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job state store
jobs: dict[str, dict[str, Any]] = {}

# Work queue to enforce single-job execution concurrency
job_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()


def cleanup_old_jobs(max_age_seconds: float = 3600.0) -> None:
    """Removes job directories and state older than 1 hour."""
    now = time.time()
    if not JOBS_DIR.exists():
        return

    for job_dir in JOBS_DIR.iterdir():
        if job_dir.is_dir():
            try:
                mtime = job_dir.stat().st_mtime
                if now - mtime > max_age_seconds:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    job_id = job_dir.name
                    if job_id in jobs and jobs[job_id].get("status") in ("done", "error"):
                        del jobs[job_id]
            except Exception:
                pass


def worker_thread_loop() -> None:
    """Sequential worker thread ensuring only one CV pipeline runs at a time."""
    while True:
        try:
            item = job_queue.get()
            if item is None:
                break

            job_id = item["job_id"]
            video_path = item["video_path"]
            alpha = item["alpha"]
            low_hz = item["low_hz"]
            high_hz = item["high_hz"]
            out_dir = item["out_dir"]

            jobs[job_id]["status"] = "running"
            jobs[job_id]["stage"] = "reading"
            jobs[job_id]["percent"] = 0.0

            def progress_callback(stage: str, percent: float) -> None:
                jobs[job_id]["stage"] = stage
                jobs[job_id]["percent"] = round(float(percent), 1)

            try:
                run_pipeline(
                    video_path=video_path,
                    checkpoint_path=str(CHECKPOINT_PATH),
                    out_dir=str(out_dir),
                    alpha=alpha,
                    low_hz=low_hz,
                    high_hz=high_hz,
                    progress_cb=progress_callback,
                    write_side_by_side=True,
                )

                jobs[job_id]["status"] = "done"
                jobs[job_id]["stage"] = "done"
                jobs[job_id]["percent"] = 100.0
                jobs[job_id]["outputs"] = {
                    "raw": f"/jobs/{job_id}/raw.mp4",
                    "baseline": f"/jobs/{job_id}/baseline.mp4",
                    "amplified": f"/jobs/{job_id}/amplified.mp4",
                    "amplified_frames": f"/jobs/{job_id}/amplified_frames.mp4",
                    "flow": f"/jobs/{job_id}/flow.mp4",
                }
            except Exception as exc:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(exc)
            finally:
                job_queue.task_done()
        except Exception:
            time.sleep(0.1)


# Start background worker daemon
_worker = threading.Thread(target=worker_thread_loop, daemon=True)
_worker.start()

# Initialize FastAPI application
app = FastAPI(
    title="Motion-Amplified Camouflage Detection API",
    description="Video camouflaged-object-detection service fusing appearance and Eulerian motion magnification",
    version="1.0.0",
)

# CORS enabled for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static video endpoints with built-in HTTP Range request support
app.mount("/jobs", StaticFiles(directory=str(JOBS_DIR)), name="jobs")
app.mount("/data/jobs", StaticFiles(directory=str(JOBS_DIR)), name="data_jobs")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    cleanup_old_jobs()
    return {"status": "ok"}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    alpha: float = Form(15.0),
    low_hz: float = Form(0.2),
    high_hz: float = Form(3.0),
):
    """
    Submits a video file for processing.
    Validates: video file format, max size 50 MB, max duration 15 seconds.
    Saves to backend/data/jobs/<job_id>/input.mp4 and enqueues background processing.
    """
    cleanup_old_jobs()

    # 1. Validate file format / mime-type
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    is_video = (file.content_type and file.content_type.startswith("video/")) or (ext in allowed_exts)
    if not is_video:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only video files (MP4, MOV, AVI, etc.) are accepted.",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.mp4"

    # 2. Stream to disk and validate size (max 50 MB)
    max_bytes = 50 * 1024 * 1024
    bytes_written = 0
    try:
        with open(input_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    out_file.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=400,
                        detail="File too large. Maximum allowed video size is 50 MB.",
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # 3. Validate video readable and max 15s duration via cv2
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Corrupted or unreadable video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    if frame_count < 2:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400, detail="Video too short; at least 2 frames are required for optical flow."
        )

    duration = frame_count / fps if fps > 0 else 0
    if duration > 15.5:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Video too long ({duration:.1f}s). Maximum allowed duration is 15 seconds.",
        )

    # 4. Initialize job state and enqueue
    jobs[job_id] = {
        "status": "queued",
        "stage": "queued",
        "percent": 0.0,
        "error": None,
        "outputs": None,
        "created_at": time.time(),
    }

    job_queue.put({
        "job_id": job_id,
        "video_path": str(input_path),
        "alpha": alpha,
        "low_hz": low_hz,
        "high_hz": high_hz,
        "out_dir": str(job_dir),
    })

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Retrieves the status and outputs of a processing job.
    Returns status: 'queued', 'running', 'done', or 'error'.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return {
        "status": job["status"],
        "stage": job["stage"],
        "percent": job["percent"],
        "error": job["error"],
        "outputs": job["outputs"],
    }


@app.get("/api/samples")
async def list_samples():
    """
    Lists precomputed sample folders in samples/ with title, thumbnail,
    and output stream mappings.
    """
    samples_list = []

    if SAMPLES_DIR.exists():
        for item in sorted(SAMPLES_DIR.iterdir()):
            if item.is_dir():
                raw_file = item / "raw.mp4"
                if raw_file.exists():
                    thumb_file = item / "thumbnail.jpg"
                    if not thumb_file.exists():
                        # Auto-extract first frame as thumbnail if absent
                        cap = cv2.VideoCapture(str(raw_file))
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            cv2.imwrite(str(thumb_file), frame)

                    samples_list.append({
                        "id": item.name,
                        "title": item.name.replace("_", " ").title(),
                        "thumbnail": f"/samples/{item.name}/thumbnail.jpg" if thumb_file.exists() else None,
                        "outputs": {
                            "raw": f"/samples/{item.name}/raw.mp4",
                            "baseline": f"/samples/{item.name}/baseline.mp4",
                            "amplified": f"/samples/{item.name}/amplified.mp4",
                            "amplified_frames": f"/samples/{item.name}/amplified_frames.mp4",
                            "flow": f"/samples/{item.name}/flow.mp4",
                        },
                    })

    return {"samples": samples_list}
