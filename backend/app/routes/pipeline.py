"""Pipeline execution endpoints."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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
        DEFAULT_HIGH_HZ,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_LOW_HZ,
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SIZE,
        SAMPLES_DIR,
        UPLOADS_DIR,
    )
    from app.services.job_runner import get_job, submit_pipeline_job
except ImportError:
    from backend.app.config import (
        DEFAULT_ALPHA,
        DEFAULT_BASE_CH,
        DEFAULT_HIGH_HZ,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_LOW_HZ,
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SIZE,
        SAMPLES_DIR,
        UPLOADS_DIR,
    )
    from backend.app.services.job_runner import get_job, submit_pipeline_job

router = APIRouter(prefix="/api", tags=["pipeline"])


@router.post("/process")
async def process_video(
    file: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
    alpha: float = Form(DEFAULT_ALPHA),
    low_hz: float = Form(DEFAULT_LOW_HZ),
    high_hz: float = Form(DEFAULT_HIGH_HZ),
    window_size: int = Form(DEFAULT_WINDOW_SIZE),
    overlap: int = Form(DEFAULT_OVERLAP),
    image_size: int = Form(DEFAULT_IMAGE_SIZE),
    base_ch: int = Form(DEFAULT_BASE_CH),
):
    """
    Submits a video for motion amplification and camouflage breaking inference.
    Accepts either an uploaded video file or a selected sample clip.
    """
    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    if file is not None and file.filename:
        ext = Path(file.filename).suffix or ".mp4"
        video_path = job_upload_dir / f"input{ext}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    elif sample_id:
        sample_path = SAMPLES_DIR / sample_id
        if not sample_path.exists():
            root_clip = REPO_ROOT / sample_id
            if root_clip.exists():
                sample_path = root_clip
            else:
                raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        video_path = sample_path
    else:
        raise HTTPException(status_code=400, detail="Must provide either a video file or sample_id.")

    job = submit_pipeline_job(
        job_id=job_id,
        video_path=str(video_path),
        alpha=alpha,
        low_hz=low_hz,
        high_hz=high_hz,
        image_size=image_size,
        base_ch=base_ch,
        window_size=window_size,
        overlap=overlap,
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "percent": job.percent,
    }


@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Returns the live execution stage and progress percentage of a job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "percent": job.percent,
        "error": job.error,
    }


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Returns the output video streams and metadata for a completed job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")

    if job.status != "completed" or not job.results:
        raise HTTPException(
            status_code=202,
            detail={"message": "Job is still processing", "status": job.status, "percent": job.percent},
        )

    return job.results
