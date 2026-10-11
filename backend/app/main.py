"""Re-export app from backend/main.py for backwards compatibility."""

from __future__ import annotations

from pathlib import Path
import sys

_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent

for p in [str(_REPO_ROOT), str(_BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from main import app
except ImportError:
    from backend.main import app
