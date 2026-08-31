"""Bug #15: upload rejections carry a stable code, not just an English string.

These are the most common failures a normal user hits — wrong file type, file
too large, corrupted media — and they used to reach a Hebrew screen as raw
English techno-strings, because the responses had no code for the UI to
translate. Every 4xx on the upload surface now keys into errors.byCode.

The 413 has TWO sources and both must speak JSON: the route's own check, and
Flask's MAX_CONTENT_LENGTH abort that fires BEFORE any route runs (its default
is an HTML page — response.json() in the frontend then throws).
"""

import io
import os
import sys

import pytest

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

pytestmark = pytest.mark.unit


@pytest.fixture
def flask_client():
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_no_file_carries_a_code(flask_client):
    response = flask_client.post("/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["code"] == "NO_FILE"


def test_wrong_extension_carries_a_code(flask_client):
    response = flask_client.post(
        "/upload",
        data={"file": (io.BytesIO(b"not a video"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_FILE_TYPE"


def test_flask_level_413_is_json_with_a_code(flask_client):
    """MAX_CONTENT_LENGTH aborts before ANY route runs, and Flask's default 413
    is an HTML page. The limit is forced tiny for the request so the abort
    fires without building a gigabyte body; what matters is that the answer is
    JSON with the stable code, whatever the configured limit is."""
    from app import app as flask_app
    from config import get_config

    original = flask_app.config["MAX_CONTENT_LENGTH"]
    flask_app.config["MAX_CONTENT_LENGTH"] = 1024  # 1KB — anything trips it
    try:
        response = flask_client.post(
            "/upload",
            data={"file": (io.BytesIO(b"\x00" * 4096), "big.mp4")},
            content_type="multipart/form-data",
        )
    finally:
        flask_app.config["MAX_CONTENT_LENGTH"] = original

    assert response.status_code == 413
    data = response.get_json()
    assert data is not None, "413 must be JSON, not Flask's HTML page"
    assert data["code"] == "FILE_TOO_LARGE"
    assert data["max_mb"] == get_config().MAX_FILE_SIZE // (1024 * 1024)


def test_unreadable_media_carries_a_code(flask_client):
    """11MB of zeros with an .mp4 name: passes the size gate, fails the probe —
    and the probe rejection already ships its own code."""
    response = flask_client.post(
        "/upload",
        data={"file": (io.BytesIO(b"\x00" * (1024 * 1024)), "junk.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["code"] == "UNSUPPORTED_MEDIA"
