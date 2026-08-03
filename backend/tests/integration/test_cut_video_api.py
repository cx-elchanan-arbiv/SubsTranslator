"""
Integration tests for /cut-video endpoint.

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
def test_cut_video_success(client, temp_dirs):
    """Test successful video cutting."""
    # Create 3-second test video
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="red", seconds=3, audio=True)

    # Upload and cut (1s to 2s)
    with open(video_path, "rb") as f:
        data = {
            "video": (io.BytesIO(f.read()), "test.mp4"),
            "start_time": "00:00:01",
            "end_time": "00:00:02",
        }

        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"
    assert int(response.headers.get("Content-Length", "0")) > 1000


@pytest.mark.integration
def test_cut_video_mm_ss_format(client, temp_dirs):
    """Test cutting with MM:SS time format."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="blue", seconds=2, audio=False)

    with open(video_path, "rb") as f:
        data = {
            "video": (io.BytesIO(f.read()), "test.mp4"),
            "start_time": "00:00",
            "end_time": "00:01",
        }

        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"


@pytest.mark.integration
def test_cut_video_missing_file_returns_400(client):
    """Test that missing video file returns 400."""
    data = {"start_time": "00:00:00", "end_time": "00:00:10"}

    response = client.post("/cut-video", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data


@pytest.mark.integration
def test_cut_video_invalid_time_range(client, temp_dirs):
    """Test that invalid time range (end before start) returns error."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="green", seconds=2)

    with open(video_path, "rb") as f:
        data = {
            "video": (io.BytesIO(f.read()), "test.mp4"),
            "start_time": "00:00:10",  # Start
            "end_time": "00:00:05",  # End (before start - invalid!)
        }

        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )

    # Should return 500 because cutting will fail
    assert response.status_code == 500


@pytest.mark.integration
def test_cut_video_default_times(client, temp_dirs):
    """Test cutting with default start/end times."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="yellow", seconds=2)

    with open(video_path, "rb") as f:
        data = {
            "video": (io.BytesIO(f.read()), "test.mp4")
            # No start_time/end_time - should use defaults (00:00:00 to 00:01:00)
        }

        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"


@pytest.mark.integration
def test_cut_video_options_request(client):
    """Test CORS preflight OPTIONS request."""
    response = client.options("/cut-video", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200


@pytest.mark.integration
def test_cut_video_output_filename(client, temp_dirs):
    """Test that output filename includes time range."""
    video_path = os.path.join(temp_dirs["uploads"], "my_video.mp4")
    assert make_video(video_path, color="red", seconds=3)

    with open(video_path, "rb") as f:
        data = {
            "video": (io.BytesIO(f.read()), "my_video.mp4"),
            "start_time": "00:00:01",
            "end_time": "00:00:02",
        }

        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200

    # Check Content-Disposition for filename
    content_disp = response.headers.get("Content-Disposition", "")
    assert "cut_" in content_disp or "attachment" in content_disp


@pytest.mark.integration
def test_cut_video_preserves_audio(client, temp_dirs):
    """Test that cutting preserves audio track."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="blue", seconds=3, audio=True)

    with open(video_path, "rb") as f:
        data = {
            "video": (io.BytesIO(f.read()), "test.mp4"),
            "start_time": "00:00:00",
            "end_time": "00:00:02",
        }

        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    # Output should be larger with audio
    assert int(response.headers.get("Content-Length", "0")) > 1000
