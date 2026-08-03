import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

import pytest
import requests

# Use relative paths that work both locally and in CI
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
BASE_URL = "http://127.0.0.1:8081"


def start_backend():
    os.chdir(str(BACKEND_ROOT))
    # Use current Python executable instead of hardcoded path
    os.execv(sys.executable, [sys.executable, "app.py"])


@pytest.fixture(scope="session", autouse=True)
def backend_server():
    p = Process(target=start_backend)
    p.daemon = True
    p.start()
    # Wait for health
    for _ in range(60):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    yield
    p.terminate()
    p.join(timeout=5)


@pytest.mark.integration
def test_health_endpoint(backend_server):
    r = requests.get(f"{BASE_URL}/health", timeout=3)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "healthy"
    assert "ffmpeg_installed" in data


@pytest.mark.integration
def test_cors_preflight_youtube(backend_server):
    r = requests.options(f"{BASE_URL}/youtube")
    assert r.status_code in (200, 204)
    # Headers presence
    assert r.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in r.headers.get("Access-Control-Allow-Methods", "")
