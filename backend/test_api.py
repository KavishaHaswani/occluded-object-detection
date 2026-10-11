"""
Unit and integration tests for FastAPI backend using TestClient.
Tests health, samples listing, range requests, validation, and job processing.
"""

import io
import time
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from main import app
except ImportError:
    from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)
    if len(data["samples"]) > 0:
        sample = data["samples"][0]
        assert "title" in sample
        assert "thumbnail" in sample
        assert "outputs" in sample
        for key in ["raw", "baseline", "amplified", "amplified_frames", "flow"]:
            assert key in sample["outputs"]


def test_static_range_support():
    sample_res = client.get("/api/samples")
    samples = sample_res.json()["samples"]
    if len(samples) > 0:
        sample_url = samples[0]["outputs"]["raw"]
        res = client.get(sample_url, headers={"Range": "bytes=0-100"})
        assert res.status_code == 206
        assert "bytes 0-100/" in res.headers.get("content-range", "")


def test_validation_rejects_non_video():
    dummy_file = io.BytesIO(b"This is not a video file.")
    response = client.post(
        "/api/jobs",
        files={"file": ("test.txt", dummy_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_validation_rejects_long_video(tmp_path):
    # Create an artificial 20-second video at 10 fps (200 frames)
    video_path = tmp_path / "long_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(200):
        out.write(frame)
    out.release()

    with open(video_path, "rb") as f:
        response = client.post(
            "/api/jobs",
            files={"file": ("long_video.mp4", f, "video/mp4")},
        )
    assert response.status_code == 400
    assert "Video too long" in response.json()["detail"]


def test_valid_job_lifecycle(tmp_path):
    # Create a short 3-second synthetic video at 10 fps (30 frames)
    video_path = tmp_path / "valid_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    for i in range(30):
        frame = np.full((64, 64, 3), (i * 8) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(video_path, "rb") as f:
        res = client.post(
            "/api/jobs",
            files={"file": ("valid_test.mp4", f, "video/mp4")},
            data={"alpha": "15.0", "low_hz": "0.2", "high_hz": "3.0"},
        )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert job_id is not None

    # Check job status immediately
    status_res = client.get(f"/api/jobs/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] in ["queued", "running", "done"]

    # Poll for completion (up to 30 seconds)
    max_wait = 30
    start = time.time()
    while time.time() - start < max_wait:
        poll_res = client.get(f"/api/jobs/{job_id}")
        data = poll_res.json()
        if data["status"] in ["done", "error"]:
            break
        time.sleep(0.5)

    final_res = client.get(f"/api/jobs/{job_id}")
    final_data = final_res.json()
    assert final_data["status"] == "done"
    assert final_data["percent"] == 100.0
    assert final_data["outputs"] is not None
    assert "raw" in final_data["outputs"]
    assert "amplified" in final_data["outputs"]
    assert "flow" in final_data["outputs"]


if __name__ == "__main__":
    print("Running test_health ...")
    test_health()
    print("Running test_samples ...")
    test_samples()
    print("Running test_static_range_support ...")
    test_static_range_support()
    print("Running test_validation_rejects_non_video ...")
    test_validation_rejects_non_video()
    print("Running test_validation_rejects_long_video ...")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_validation_rejects_long_video(Path(td))
    print("Running test_valid_job_lifecycle ...")
    with tempfile.TemporaryDirectory() as td:
        test_valid_job_lifecycle(Path(td))
    print("ALL TESTS PASSED SUCCESSFULLY!")
