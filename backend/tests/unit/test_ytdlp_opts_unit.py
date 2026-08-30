"""The yt-dlp option dicts both download paths hand to YoutubeDL.

These options are the download's entire behaviour — format selection, timeouts, retries,
the single-video guard. A value hardcoded back into the dict instead of read from config
is invisible in review and only shows up as "why did we get 360p again".

The file previously carried no ``unit`` mark, so CI never ran it, and ``last_opts`` was a
class attribute shared across both tests — the second test could pass on the first one's
leftovers.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

pytestmark = pytest.mark.unit


class DummyYDL:
    """Dummy YoutubeDL for capturing options."""

    last_opts = None

    def __init__(self, opts):
        DummyYDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=True):
        return {
            "title": "Unit Video",
            "duration": 1,
            "duration_string": "00:00:01",
            "view_count": 0,
            "upload_date": "20250101",
            "uploader": "unit",
            "thumbnail": "",
            "description": "",
            "width": 640,
            "height": 360,
            "fps": 30,
            "filesize": 0,
        }

    def prepare_filename(self, info):
        return os.path.join("/tmp/test_downloads", f"{info.get('title','Unit')}.mp4")


def test_download_youtube_video_builds_opts(monkeypatch, tmp_path):
    """``download_youtube_video`` reads its options from config, not from literals."""
    from config import get_config
    from services import youtube_service

    DummyYDL.last_opts = None

    downloads_folder = str(tmp_path)
    os.makedirs(downloads_folder, exist_ok=True)

    monkeypatch.setattr(youtube_service, "DOWNLOADS_FOLDER", downloads_folder)
    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", DummyYDL)

    real_config = get_config()
    mock_config = MagicMock(
        USE_FAKE_YTDLP=False,
        DOWNLOADS_FOLDER=downloads_folder,
        FAST_WORK_DIR=downloads_folder,
        YTDLP_OPTIMIZED_FORMAT=real_config.YTDLP_OPTIMIZED_FORMAT,
        YTDLP_SOCKET_TIMEOUT=real_config.YTDLP_SOCKET_TIMEOUT,
        YTDLP_FRAGMENT_RETRIES=real_config.YTDLP_FRAGMENT_RETRIES,
        YTDLP_RETRIES=real_config.YTDLP_RETRIES,
        YTDLP_EXTRACTOR_ARGS=real_config.YTDLP_EXTRACTOR_ARGS,
        DEBUG=False,
    )
    monkeypatch.setattr(youtube_service, "config", mock_config)

    youtube_service.download_youtube_video("http://example.com/video", "high")

    opts = DummyYDL.last_opts
    assert opts is not None, "YoutubeDL was not called"

    # Values, not just key presence — the old version passed with format="".
    assert opts["format"] == real_config.YTDLP_OPTIMIZED_FORMAT
    assert opts["socket_timeout"] == real_config.YTDLP_SOCKET_TIMEOUT
    assert opts["fragment_retries"] == real_config.YTDLP_FRAGMENT_RETRIES
    assert opts["retries"] == real_config.YTDLP_RETRIES
    assert opts["merge_output_format"] == "mp4"

    # A playlist URL must yield one video, never the whole playlist.
    assert opts["noplaylist"] is True

    # The player client comes from config, never from a literal here — which client
    # works changes with whatever YouTube is enforcing that month.
    assert opts["extractor_args"] == real_config.YTDLP_EXTRACTOR_ARGS

    # Remux-only: faststart and nothing else, so no re-encode creeps back in.
    assert opts["postprocessor_args"]["ffmpeg"] == ["-movflags", "+faststart"]


def test_download_youtube_video_with_progress_builds_opts(monkeypatch, tmp_path):
    """Test that download_youtube_video_with_progress builds correct yt-dlp options."""
    from config import get_config
    from services import youtube_service

    DummyYDL.last_opts = None

    # Create temp downloads folder
    downloads_folder = str(tmp_path)
    os.makedirs(downloads_folder, exist_ok=True)

    # Patch DOWNLOADS_FOLDER and yt_dlp
    monkeypatch.setattr(youtube_service, "DOWNLOADS_FOLDER", downloads_folder)
    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", DummyYDL)

    # Get real config for expected values
    real_config = get_config()

    # Create mock config with real values
    # Note: download_youtube_video_with_progress uses YTDLP_OPTIMIZED_FORMAT
    mock_config = MagicMock(
        USE_FAKE_YTDLP=False,
        DOWNLOADS_FOLDER=downloads_folder,
        FAST_WORK_DIR=downloads_folder,  # Used by download_youtube_video_with_progress
        YTDLP_OPTIMIZED_FORMAT=real_config.YTDLP_OPTIMIZED_FORMAT,  # Phase A format
        YTDLP_SOCKET_TIMEOUT=real_config.YTDLP_SOCKET_TIMEOUT,
        YTDLP_FRAGMENT_RETRIES=real_config.YTDLP_FRAGMENT_RETRIES,
        YTDLP_RETRIES=real_config.YTDLP_RETRIES,
        YTDLP_CACHE_DIR=real_config.YTDLP_CACHE_DIR,
        YTDLP_MERGE_OUTPUT_FORMAT=real_config.YTDLP_MERGE_OUTPUT_FORMAT,
        YTDLP_RESTRICT_FILENAMES=real_config.YTDLP_RESTRICT_FILENAMES,
        YTDLP_CONTINUE_DL=real_config.YTDLP_CONTINUE_DL,
        DEBUG=False,
    )
    monkeypatch.setattr(youtube_service, "config", mock_config)

    # Simple progress manager stub
    class PM:
        def __init__(self):
            self.steps = [{"progress": 0}]

        def set_step_progress(self, *_args, **_kwargs):
            pass

        def log(self, *_args, **_kwargs):
            pass

    filename, meta = youtube_service.download_youtube_video_with_progress(
        "http://example.com/video", "medium", PM()
    )

    opts = DummyYDL.last_opts
    assert opts is not None, "YoutubeDL was not called"

    # Verify config values are used (YTDLP_OPTIMIZED_FORMAT is the Phase A format)
    assert opts["format"] == real_config.YTDLP_OPTIMIZED_FORMAT
    assert opts["socket_timeout"] == real_config.YTDLP_SOCKET_TIMEOUT
    assert opts["fragment_retries"] == real_config.YTDLP_FRAGMENT_RETRIES
    assert opts["retries"] == real_config.YTDLP_RETRIES
    assert opts["merge_output_format"] == real_config.YTDLP_MERGE_OUTPUT_FORMAT
    assert opts["restrict_filenames"] == real_config.YTDLP_RESTRICT_FILENAMES
    assert opts["continue_dl"] == real_config.YTDLP_CONTINUE_DL
