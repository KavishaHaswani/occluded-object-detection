"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root and backend directory are on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent  # app
_BACKEND_DIR = _CURRENT_DIR.parent             # backend
_REPO_ROOT = _BACKEND_DIR.parent               # final_Project

for p in [str(_REPO_ROOT), str(_BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from app.config import MEDIA_DIR, SAMPLES_DIR
    from app.routes.pipeline import router as pipeline_router
    from app.routes.samples import router as samples_router
except ImportError:
    from backend.app.config import MEDIA_DIR, SAMPLES_DIR
    from backend.app.routes.pipeline import router as pipeline_router
    from backend.app.routes.samples import router as samples_router

app = FastAPI(
    title="Motion-Amplified Camouflage Breaking API",
    description="Backend API for two-stream Eulerian motion-amplified camouflaged object detection",
    version="1.0.0",
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static video directories
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

# Include routers
app.include_router(pipeline_router)
app.include_router(samples_router)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "motion-amplified-cod-api",
    }
