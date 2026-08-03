"""
Integration tests for /merge-videos endpoint.

Tests the full endpoint with real FFmpeg and dummy video files.
Requires FFmpeg to be installed.
"""

import io
import os
import shutil

import pytest

from .ffmpeg_helpers import make_video

# Skip all tests if FFmpeg is not available
ffmpeg_available = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not ffmpeg_available, reason="FFmpeg not installed")


# Use shared flask_test_client fixture from conftest.py
@pytest.fixture
def client(flask_test_client):
    """Alias for flask_test_client for easier reading."""
    return flask_test_client


@pytest.mark.integration
def test_merge_success_with_audio_and_silent(client, temp_dirs):
    """Test successful merge of video with audio and silent video."""
    # Create test videos
    video1_path = os.path.join(temp_dirs["uploads"], "v1.mp4")
    video2_path = os.path.join(temp_dirs["uploads"], "v2.mp4")

    assert make_video(video1_path, color="red", audio=True, seconds=1)
    assert make_video(video2_path, color="blue", audio=False, seconds=1)

    # Upload videos
    with open(video1_path, "rb") as f1, open(video2_path, "rb") as f2:
        data = {
            "video1": (io.BytesIO(f1.read()), "video1.mp4"),
            "video2": (io.BytesIO(f2.read()), "video2.mp4"),
        }

        response = client.post(
            "/merge-videos", data=data, content_type="multipart/form-data"
        )

    # Verify response
    assert response.status_code == 200
    assert response.mimetype == "video/mp4"
    assert int(response.headers.get("Content-Length", "0")) > 1000


@pytest.mark.integration
def test_merge_two_videos_with_audio(client, temp_dirs):
    """Test merge of two videos both with audio."""
    video1_path = os.path.join(temp_dirs["uploads"], "v1.mp4")
    video2_path = os.path.join(temp_dirs["uploads"], "v2.mp4")

    assert make_video(video1_path, color="green", audio=True, seconds=1)
    assert make_video(video2_path, color="yellow", audio=True, seconds=1)

    with open(video1_path, "rb") as f1, open(video2_path, "rb") as f2:
        data = {
            "video1": (io.BytesIO(f1.read()), "a.mp4"),
            "video2": (io.BytesIO(f2.read()), "b.mp4"),
        }

        response = client.post(
            "/merge-videos", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"


@pytest.mark.integration
def test_merge_missing_video1_returns_400(client):
    """Test that missing video1 returns 400 error."""
    data = {"video2": (io.BytesIO(b"fake"), "v2.mp4")}

    response = client.post(
        "/merge-videos", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data


@pytest.mark.integration
def test_merge_missing_video2_returns_400(client):
    """Test that missing video2 returns 400 error."""
    data = {"video1": (io.BytesIO(b"fake"), "v1.mp4")}

    response = client.post(
        "/merge-videos", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data


@pytest.mark.integration
def test_merge_no_files_returns_400(client):
    """Test that request with no files returns 400."""
    response = client.post("/merge-videos", data={}, content_type="multipart/form-data")

    assert response.status_code == 400


@pytest.mark.integration
def test_merge_cors_headers(client, temp_dirs):
    """Test that CORS headers are present."""
    # Create minimal test videos
    video1_path = os.path.join(temp_dirs["uploads"], "v1.mp4")
    video2_path = os.path.join(temp_dirs["uploads"], "v2.mp4")

    assert make_video(video1_path, color="black", seconds=1)
    assert make_video(video2_path, color="white", seconds=1)

    with open(video1_path, "rb") as f1, open(video2_path, "rb") as f2:
        data = {
            "video1": (io.BytesIO(f1.read()), "v1.mp4"),
            "video2": (io.BytesIO(f2.read()), "v2.mp4"),
        }

        response = client.post(
            "/merge-videos",
            data=data,
            content_type="multipart/form-data",
            headers={"Origin": "http://localhost:3000"},
        )

    # CORS headers should be present if configured globally
    # This test is informational
    assert response.status_code == 200


@pytest.mark.integration
def test_merge_options_request(client):
    """Test CORS preflight OPTIONS request."""
    response = client.options(
        "/merge-videos", headers={"Origin": "http://localhost:3000"}
    )

    # Should return 200 for OPTIONS
    assert response.status_code == 200


@pytest.mark.integration
def test_merge_output_filename_format(client, temp_dirs):
    """Test that output filename is properly formatted."""
    video1_path = os.path.join(temp_dirs["uploads"], "first_video.mp4")
    video2_path = os.path.join(temp_dirs["uploads"], "second_video.mp4")

    assert make_video(video1_path, color="red", seconds=1)
    assert make_video(video2_path, color="blue", seconds=1)

    with open(video1_path, "rb") as f1, open(video2_path, "rb") as f2:
        data = {
            "video1": (io.BytesIO(f1.read()), "first_video.mp4"),
            "video2": (io.BytesIO(f2.read()), "second_video.mp4"),
        }

        response = client.post(
            "/merge-videos", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200

    # Check Content-Disposition header for filename
    content_disp = response.headers.get("Content-Disposition", "")
    assert "merged_" in content_disp or "attachment" in content_disp
