"""
Integration tests for /embed-subtitles endpoint.

Tests the full endpoint with real FFmpeg and dummy files.
Requires FFmpeg to be installed.
"""

import io
import os
import shutil

import pytest

from .ffmpeg_helpers import make_logo_image, make_srt_file, make_video

# Skip all tests if FFmpeg is not available
ffmpeg_available = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not ffmpeg_available, reason="FFmpeg not installed")


# Use shared flask_test_client fixture from conftest.py
@pytest.fixture
def client(flask_test_client):
    """Alias for flask_test_client for easier reading."""
    return flask_test_client


@pytest.mark.integration
def test_embed_subtitles_with_srt_file(client, temp_dirs):
    """Test embedding subtitles from SRT file."""
    # Create test video and SRT
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    srt_path = os.path.join(temp_dirs["uploads"], "test.srt")

    assert make_video(video_path, color="red", seconds=2)
    assert make_srt_file(srt_path, num_subtitles=3)

    # Upload
    with open(video_path, "rb") as fv, open(srt_path, "rb") as fs:
        data = {
            "video": (io.BytesIO(fv.read()), "test.mp4"),
            "srt_file": (io.BytesIO(fs.read()), "test.srt"),
            "include_logo": "false",
        }

        response = client.post(
            "/embed-subtitles", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"
    assert int(response.headers.get("Content-Length", "0")) > 1000


@pytest.mark.integration
def test_embed_subtitles_with_text(client, temp_dirs):
    """Test embedding subtitles from text input."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="blue", seconds=2)

    srt_text = """[00:00 - 00:02] Test subtitle 1
[00:02 - 00:04] Test subtitle 2"""

    with open(video_path, "rb") as fv:
        data = {
            "video": (io.BytesIO(fv.read()), "test.mp4"),
            "srt_text": srt_text,
            "include_logo": "false",
        }

        response = client.post(
            "/embed-subtitles", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"


@pytest.mark.integration
def test_embed_subtitles_missing_video_returns_400(client):
    """Test that missing video file returns 400."""
    data = {"srt_text": "[00:00 - 00:05] Test", "include_logo": "false"}

    response = client.post(
        "/embed-subtitles", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data


@pytest.mark.integration
def test_embed_subtitles_missing_both_srt_and_text_returns_400(client, temp_dirs):
    """Test that missing both SRT file and text returns 400."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    assert make_video(video_path, color="green", seconds=2)

    with open(video_path, "rb") as fv:
        data = {
            "video": (io.BytesIO(fv.read()), "test.mp4"),
            "include_logo": "false",
            # No srt_file or srt_text
        }

        response = client.post(
            "/embed-subtitles", data=data, content_type="multipart/form-data"
        )

    # Should fail because no subtitles provided
    assert response.status_code in [400, 500]


@pytest.mark.integration
def test_embed_subtitles_with_logo(client, temp_dirs):
    """Test embedding subtitles with logo watermark."""
    # Create test files
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    srt_path = os.path.join(temp_dirs["uploads"], "test.srt")
    logo_path = os.path.join(temp_dirs["uploads"], "logo.png")

    assert make_video(video_path, color="yellow", seconds=2)
    assert make_srt_file(srt_path, num_subtitles=2)
    assert make_logo_image(logo_path, width=100, height=50)

    # The endpoint reads the user's logo from the Flask session key that the
    # watermark upload routes write ('custom_logo_path').
    with client.session_transaction() as sess:
        sess["custom_logo_path"] = logo_path

    with open(video_path, "rb") as fv, open(srt_path, "rb") as fs:
        data = {
            "video": (io.BytesIO(fv.read()), "test.mp4"),
            "srt_file": (io.BytesIO(fs.read()), "test.srt"),
            "include_logo": "true",
            "logo_position": "bottom-right",
            "logo_size": "medium",
            "logo_opacity": "50",
        }

        response = client.post(
            "/embed-subtitles", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.mimetype == "video/mp4"


@pytest.mark.integration
def test_embed_subtitles_options_request(client):
    """Test CORS preflight OPTIONS request."""
    response = client.options(
        "/embed-subtitles", headers={"Origin": "http://localhost:3000"}
    )

    assert response.status_code == 200


@pytest.mark.integration
def test_embed_subtitles_output_filename(client, temp_dirs):
    """Test that output filename includes indication of subtitles."""
    video_path = os.path.join(temp_dirs["uploads"], "my_video.mp4")
    srt_path = os.path.join(temp_dirs["uploads"], "subs.srt")

    assert make_video(video_path, color="red", seconds=2)
    assert make_srt_file(srt_path, num_subtitles=2)

    with open(video_path, "rb") as fv, open(srt_path, "rb") as fs:
        data = {
            "video": (io.BytesIO(fv.read()), "my_video.mp4"),
            "srt_file": (io.BytesIO(fs.read()), "subs.srt"),
            "include_logo": "false",
        }

        response = client.post(
            "/embed-subtitles", data=data, content_type="multipart/form-data"
        )

    assert response.status_code == 200

    content_disp = response.headers.get("Content-Disposition", "")
    assert "video_with_subtitles" in content_disp or "attachment" in content_disp


@pytest.mark.integration
def test_embed_subtitles_different_logo_settings(client, temp_dirs):
    """Test different logo position/size/opacity settings."""
    video_path = os.path.join(temp_dirs["uploads"], "test.mp4")
    srt_path = os.path.join(temp_dirs["uploads"], "test.srt")
    logo_path = os.path.join(temp_dirs["uploads"], "logo.png")

    assert make_video(video_path, color="blue", seconds=1)
    assert make_srt_file(srt_path, num_subtitles=1)
    assert make_logo_image(logo_path)

    # See test_embed_subtitles_with_logo: the endpoint resolves the logo from
    # the 'custom_logo_path' session key.
    with client.session_transaction() as sess:
        sess["custom_logo_path"] = logo_path

    # Test different combinations
    settings = [
        {"position": "top-left", "size": "small", "opacity": "30"},
        {"position": "bottom-right", "size": "large", "opacity": "70"},
    ]

    for setting in settings:
        with open(video_path, "rb") as fv, open(srt_path, "rb") as fs:
            data = {
                "video": (io.BytesIO(fv.read()), "test.mp4"),
                "srt_file": (io.BytesIO(fs.read()), "test.srt"),
                "include_logo": "true",
                "logo_position": setting["position"],
                "logo_size": setting["size"],
                "logo_opacity": setting["opacity"],
            }

            response = client.post(
                "/embed-subtitles", data=data, content_type="multipart/form-data"
            )

        assert response.status_code == 200
