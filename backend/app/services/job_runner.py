"""Background execution engine for pipeline inference jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
import time
from typing import Any

# Ensure path bootstrap
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.config import (
        DEFAULT_ALPHA,
        DEFAULT_BASE_CH,
        DEFAULT_CHECKPOINT,
        DEFAULT_HIGH_HZ,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_LOW_HZ,
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SIZE,
        MEDIA_DIR,
    )
except ImportError:
    from backend.app.config import (
        DEFAULT_ALPHA,
        DEFAULT_BASE_CH,
        DEFAULT_CHECKPOINT,
        DEFAULT_HIGH_HZ,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_LOW_HZ,
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SIZE,
        MEDIA_DIR,
    )

# Root import made possible via sys.path bootstrap
from infer_demo import run_pipeline


@dataclass
class Job:
    job_id: str
    status: str = "pending"  # "pending", "running", "completed", "failed"
    stage: str = "queued"     # "reading", "amplifying", "inference"
    percent: float = 0.0
    error: str | None = None
    results: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def register_job(job_id: str) -> Job:
    job = Job(job_id=job_id)
    with _LOCK:
        _JOBS[job_id] = job
    return job


def _execute_pipeline_task(
    job_id: str,
    video_path: str,
    checkpoint_path: str,
    alpha: float,
    low_hz: float,
    high_hz: float,
    image_size: int,
    base_ch: int,
    window_size: int,
    overlap: int,
) -> None:
    job = get_job(job_id)
    if not job:
        return

    job.status = "running"
    job.stage = "reading"
    job.percent = 0.0

    def progress_callback(stage: str, percent: float) -> None:
        job.stage = stage
        job.percent = round(percent, 1)

    out_dir = MEDIA_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_res = run_pipeline(
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            out_dir=str(out_dir),
            alpha=alpha,
            low_hz=low_hz,
            high_hz=high_hz,
            image_size=image_size,
            base_ch=base_ch,
            window_size=window_size,
            overlap=overlap,
            progress_cb=progress_callback,
            write_side_by_side=True,
        )

        urls = {
            "raw": f"/media/{job_id}/raw.mp4",
            "baseline": f"/media/{job_id}/baseline.mp4",
            "amplified": f"/media/{job_id}/amplified.mp4",
            "amplified_frames": f"/media/{job_id}/amplified_frames.mp4",
            "flow": f"/media/{job_id}/flow.mp4",
            "demo_output": f"/media/{job_id}/demo_output.mp4",
        }

        job.results = {
            "job_id": job_id,
            "videos": urls,
            "metadata": {
                "frame_count": raw_res["frame_count"],
                "fps": raw_res["fps"],
                "width": raw_res["width"],
                "height": raw_res["height"],
            },
            "parameters": {
                "alpha": alpha,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "window_size": window_size,
                "overlap": overlap,
            },
        }
        job.status = "completed"
        job.percent = 100.0
    except Exception as e:
        job.status = "failed"
        job.error = str(e)


def submit_pipeline_job(
    job_id: str,
    video_path: str,
    checkpoint_path: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    image_size: int = DEFAULT_IMAGE_SIZE,
    base_ch: int = DEFAULT_BASE_CH,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Job:
    job = register_job(job_id)
    cp = checkpoint_path or str(DEFAULT_CHECKPOINT)

    _EXECUTOR.submit(
        _execute_pipeline_task,
        job_id=job_id,
        video_path=video_path,
        checkpoint_path=cp,
        alpha=alpha,
        low_hz=low_hz,
        high_hz=high_hz,
        image_size=image_size,
        base_ch=base_ch,
        window_size=window_size,
        overlap=overlap,
    )
    return job
