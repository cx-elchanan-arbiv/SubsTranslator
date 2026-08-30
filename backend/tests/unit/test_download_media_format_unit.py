"""The MP3 option on the download-only path.

Everything here is about one thing: ``media_format="mp3"`` must change the yt-dlp
options in exactly the ways audio needs, and ``media_format="mp4"`` (or no argument
at all) must leave the video path byte-for-byte as it was. The traps this file pins
down are the ones that fail silently rather than loudly:

* ``+faststart`` is an MP4 container flag. Passing it on an MP3 write makes ffmpeg
  fail *after* a completed download.
* ``merge_output_format`` asks for an MP4 container; there is no second stream to
  merge when only audio was requested.
* ``prepare_filename`` predicts the name from the FORMAT, before postprocessing.
  Audio extraction downloads ``.m4a`` and writes ``.mp3``, so the predicted path
  does not exist and the move block would skip — reporting "no file" for a download
  that actually succeeded.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Must be set BEFORE `app` is imported by the route tests below: `app` starts the token
# cleanup scheduler, whose non-daemon Timer is skipped only when TESTING is exactly
# "true" — otherwise pytest hangs at shutdown. Same reasoning as
# tests/unit/test_subtitle_flags_plumbing.py.
os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

pytestmark = pytest.mark.unit


class RecordingYDL:
    """Captures the options dict and pretends the download produced a file.

    ``extract_info`` reports the pre-postprocessing extension the way yt-dlp does,
    and writes the *post*-postprocessing file, so the filename resolution under test
    is exercised rather than assumed.
    """

    last_opts = None
    work_dir = None
    downloaded_ext = ".m4a"
    written_ext = ".mp3"
    report_filepath = True

    def __init__(self, opts):
        RecordingYDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=True):
        written = os.path.join(RecordingYDL.work_dir, f"Unit Video{self.written_ext}")
        with open(written, "wb") as handle:
            handle.write(b"audio")
        info = {
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
        if self.report_filepath:
            info["requested_downloads"] = [{"filepath": written}]
        return info

    def prepare_filename(self, info):
        # The name yt-dlp predicts before the postprocessor runs.
        return os.path.join(
            RecordingYDL.work_dir, f"{info.get('title', 'Unit')}{self.downloaded_ext}"
        )


def _patched_service(monkeypatch, tmp_path):
    from config import get_config
    from services import youtube_service

    RecordingYDL.last_opts = None
    RecordingYDL.work_dir = str(tmp_path)
    RecordingYDL.report_filepath = True

    monkeypatch.setattr(youtube_service, "DOWNLOADS_FOLDER", str(tmp_path))
    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", RecordingYDL)

    real_config = get_config()
    monkeypatch.setattr(
        youtube_service,
        "config",
        MagicMock(
            USE_FAKE_YTDLP=False,
            DOWNLOADS_FOLDER=str(tmp_path),
            FAST_WORK_DIR=str(tmp_path),
            YTDLP_OPTIMIZED_FORMAT=real_config.YTDLP_OPTIMIZED_FORMAT,
            YTDLP_SOCKET_TIMEOUT=real_config.YTDLP_SOCKET_TIMEOUT,
            YTDLP_FRAGMENT_RETRIES=real_config.YTDLP_FRAGMENT_RETRIES,
            YTDLP_RETRIES=real_config.YTDLP_RETRIES,
            YTDLP_CACHE_DIR=real_config.YTDLP_CACHE_DIR,
            YTDLP_MERGE_OUTPUT_FORMAT=real_config.YTDLP_MERGE_OUTPUT_FORMAT,
            YTDLP_RESTRICT_FILENAMES=real_config.YTDLP_RESTRICT_FILENAMES,
            YTDLP_CONTINUE_DL=real_config.YTDLP_CONTINUE_DL,
            DEBUG=False,
        ),
    )
    return youtube_service


class _PM:
    def __init__(self):
        self.steps = [{"progress": 0}]

    def set_step_progress(self, *_args, **_kwargs):
        pass

    def log(self, *_args, **_kwargs):
        pass


def test_mp3_selects_audio_format_and_extractor(monkeypatch, tmp_path):
    service = _patched_service(monkeypatch, tmp_path)

    path, _meta = service.download_youtube_video_with_progress(
        "http://example.com/video", "high", _PM(), media_format="mp3"
    )

    opts = RecordingYDL.last_opts
    assert opts["format"] == service.AUDIO_ONLY_FORMAT
    assert [pp["key"] for pp in opts["postprocessors"]] == ["FFmpegExtractAudio"]
    assert opts["postprocessors"][0]["preferredcodec"] == "mp3"

    # The two video-only settings that break an MP3 write.
    assert "merge_output_format" not in opts
    assert "+faststart" not in opts["postprocessor_args"]["ffmpeg"]

    # And the download is reported at its real, post-conversion path.
    assert path.endswith(".mp3")
    assert os.path.exists(path)


def test_mp3_falls_back_to_extension_swap_when_ytdlp_reports_no_path(
    monkeypatch, tmp_path
):
    """Older yt-dlp builds omit ``requested_downloads``; the file still exists."""
    service = _patched_service(monkeypatch, tmp_path)
    RecordingYDL.report_filepath = False

    path, _meta = service.download_youtube_video_with_progress(
        "http://example.com/video", "high", _PM(), media_format="mp3"
    )

    assert path.endswith(".mp3")
    assert os.path.exists(path)


def test_mp4_path_is_unchanged(monkeypatch, tmp_path):
    """The default must not inherit anything from the audio branch."""
    from config import get_config

    service = _patched_service(monkeypatch, tmp_path)
    RecordingYDL.downloaded_ext = ".mp4"
    RecordingYDL.written_ext = ".mp4"
    try:
        service.download_youtube_video_with_progress(
            "http://example.com/video", "high", _PM()
        )
    finally:
        RecordingYDL.downloaded_ext = ".m4a"
        RecordingYDL.written_ext = ".mp3"

    opts = RecordingYDL.last_opts
    real_config = get_config()
    assert opts["format"] == real_config.YTDLP_OPTIMIZED_FORMAT
    assert opts["merge_output_format"] == real_config.YTDLP_MERGE_OUTPUT_FORMAT
    assert opts["postprocessor_args"]["ffmpeg"] == ["-movflags", "+faststart"]
    assert "postprocessors" not in opts


def test_unknown_media_format_behaves_like_mp4(monkeypatch, tmp_path):
    service = _patched_service(monkeypatch, tmp_path)
    RecordingYDL.downloaded_ext = ".mp4"
    RecordingYDL.written_ext = ".mp4"
    try:
        service.download_youtube_video_with_progress(
            "http://example.com/video", "high", _PM(), media_format="wav"
        )
    finally:
        RecordingYDL.downloaded_ext = ".m4a"
        RecordingYDL.written_ext = ".mp3"

    assert "postprocessors" not in RecordingYDL.last_opts


def test_mp3_time_range_does_not_force_stream_copy(monkeypatch, tmp_path):
    """ "-c copy" would override the codec FFmpegExtractAudio picked."""
    service = _patched_service(monkeypatch, tmp_path)

    service.download_youtube_video_with_progress(
        "http://example.com/video",
        "high",
        _PM(),
        start_time="00:00:01",
        end_time="00:00:05",
        media_format="mp3",
    )

    ffmpeg_args = RecordingYDL.last_opts["postprocessor_args"]["ffmpeg"]
    assert ffmpeg_args == ["-ss", "1", "-to", "5"]


# =====================================================================================
# The route: what actually reaches the Celery task
# =====================================================================================


@pytest.fixture
def flask_client():
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def download_task():
    """Capture apply_async without a broker."""
    from api import video_routes

    task = MagicMock()
    task.apply_async.return_value = MagicMock(id="task-123")
    with patch.object(video_routes, "download_youtube_only_task", task):
        yield task


def _post(client, body):
    return client.post("/download-video-only", json=body)


URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"url": URL}, "mp4"),  # a client that predates the option
        ({"url": URL, "media_format": "mp4"}, "mp4"),
        ({"url": URL, "media_format": "mp3"}, "mp3"),
        ({"url": URL, "media_format": "MP3"}, "mp3"),
        ({"url": URL, "media_format": None}, "mp4"),
        ({"url": URL, "media_format": "wav"}, "mp4"),  # never reaches yt-dlp
        ({"url": URL, "media_format": "../../etc/passwd"}, "mp4"),
    ],
)
def test_route_passes_only_a_validated_format(
    flask_client, download_task, body, expected
):
    response = _post(flask_client, body)
    assert response.status_code == 202, response.get_data(as_text=True)

    args = download_task.apply_async.call_args.kwargs["args"]
    assert args[0] == URL
    assert args[-1] == expected
