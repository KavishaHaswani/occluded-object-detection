"""Configuration and path resolutions for the FastAPI backend."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root and backend directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Paths
DEFAULT_CHECKPOINT = REPO_ROOT / "best_model.zip"
MEDIA_DIR = REPO_ROOT / "backend" / "media"
SAMPLES_DIR = REPO_ROOT / "samples"
UPLOADS_DIR = MEDIA_DIR / "uploads"

# Create required directories
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Default Pipeline Hyperparameters
DEFAULT_ALPHA = 15.0
DEFAULT_LOW_HZ = 0.2
DEFAULT_HIGH_HZ = 3.0
DEFAULT_IMAGE_SIZE = 352
DEFAULT_BASE_CH = 32
DEFAULT_WINDOW_SIZE = 60
DEFAULT_OVERLAP = 30
