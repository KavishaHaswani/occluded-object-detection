"""Samples discovery endpoints."""

from __future__ import annotations

from pathlib import Path
import sys
from fastapi import APIRouter

# Ensure path bootstrap
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.config import REPO_ROOT, SAMPLES_DIR
except ImportError:
    from backend.app.config import REPO_ROOT, SAMPLES_DIR

router = APIRouter(prefix="/api/samples", tags=["samples"])


@router.get("")
async def list_samples():
    """Lists available pre-recorded demo video clips."""
    samples = []

    for f in SAMPLES_DIR.glob("*.mp4"):
        samples.append({
            "id": f.name,
            "title": f.stem.replace("_", " ").title(),
            "path": f.name,
            "is_precomputed": (SAMPLES_DIR / f.stem).is_dir(),
        })

    root_clip = REPO_ROOT / "clip.mp4"
    if root_clip.exists() and not any(s["id"] == "clip.mp4" for s in samples):
        samples.insert(0, {
            "id": "clip.mp4",
            "title": "Chameleon (Camouflaged Real Clip)",
            "path": "clip.mp4",
            "is_precomputed": False,
        })

    return {"samples": samples}
