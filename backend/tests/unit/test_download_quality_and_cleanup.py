"""The download-quality choice and the files a download leaves behind.

* quality: "fast" (720p) is the default — the owner's choice, because downloads from
  video sites reach this machine at ~0.4 MB/s and a 720p file is ~40% smaller. The
  ``quality`` argument existed before but nothing read it.
* every file a download writes in the working folder starts with the job's tag, so a
  failed download removes exactly its own leftovers (the CNN run of 2026-10-01 left
  424 MB there until the daily wipe);
* the download-only file carries the tag too: 720p and 1080p of one video no longer
  share a name;
* the daily sweep of the working folder deletes only abandoned files, not a download
  that is still running.
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pytest
import yt_dlp

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.download_quality import (  # noqa: E402
    DEFAULT_QUALITY,
    format_for_quality,
    parse_quality,
)

pytestmark = pytest.mark.unit

JOB_ID = "abcdef12-3456-7890-abcd-ef1234567890"
TAG = JOB_ID[:8]


# --------------------------------------------------------------------------- quality


class TestQuality:
    def test_the_default_is_fast(self):
        assert DEFAULT_QUALITY == "fast"
        assert parse_quality(None) == "fast"

    @pytest.mark.parametrize("value", ["high", "HIGH", " high "])
    def test_high_is_high(self, value):
        assert parse_quality(value) == "high"

    @pytest.mark.parametrize("value", ["medium", "low", "", "1080", 1080, ["high"]])
    def test_anything_else_is_the_default(self, value):
        """Legacy values ("medium" was the old signature default) and junk."""
        assert parse_quality(value) == "fast"

    def test_fast_caps_at_720(self):
        fmt = format_for_quality("fast")
        assert "height<=720" in fmt and "height<=1080" not in fmt

    def test_high_is_the_configured_format(self):
        from config import get_config

        assert format_for_quality("high") == get_config().YTDLP_OPTIMIZED_FORMAT


# --------------------------------------------------------------------------- downloads


class FailingYDL:
    """Writes partial files the way yt-dlp does, then fails mid-download."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        return {"id": "v", "title": "Clip", "ext": "mp4"}

    def process_ie_result(self, info, download=True):
        base = self.opts["outtmpl"].replace("%(title).100B.%(ext)s", "Clip")
        for suffix in (".f137.mp4.part", ".f137.mp4.part-Frag3.part", ".f137.mp4.ytdl"):
            with open(base + suffix, "wb") as handle:
                handle.write(b"partial")
        raise yt_dlp.utils.DownloadError("connection reset mid-download")


class WorkingYDL(FailingYDL):
    def process_ie_result(self, info, download=True):
        path = self.opts["outtmpl"].replace("%(title).100B.%(ext)s", "Clip.mp4")
        with open(path, "wb") as handle:
            handle.write(b"video")
        return {**info, "requested_downloads": [{"filepath": path}]}

    def prepare_filename(self, info):
        return info["requested_downloads"][0]["filepath"]


@pytest.fixture
def service(monkeypatch, tmp_path):
    from config import get_config
    from services import youtube_service

    work, final = tmp_path / "work", tmp_path / "final"
    work.mkdir()
    final.mkdir()
    real = get_config()
    monkeypatch.setattr(youtube_service, "DOWNLOADS_FOLDER", str(final))
    monkeypatch.setattr(
        youtube_service,
        "config",
        MagicMock(
            USE_FAKE_YTDLP=False,
            DOWNLOADS_FOLDER=str(final),
            FAST_WORK_DIR=str(work),
            YTDLP_OPTIMIZED_FORMAT=real.YTDLP_OPTIMIZED_FORMAT,
            YTDLP_MERGE_OUTPUT_FORMAT=real.YTDLP_MERGE_OUTPUT_FORMAT,
            DEBUG=False,
        ),
    )
    youtube_service._work, youtube_service._final = work, final
    return youtube_service


@pytest.mark.parametrize(
    "download", ["download_youtube_video", "download_youtube_video_with_progress"]
)
def test_a_failed_download_removes_its_own_leftovers_only(
    service, monkeypatch, download
):
    other_job = service._work / "99999999_Other.f137.mp4.part"
    other_job.write_bytes(b"someone else's download")
    monkeypatch.setattr(service.yt_dlp, "YoutubeDL", FailingYDL)

    with pytest.raises(Exception):
        getattr(service, download)("https://example.com/v", job_id=JOB_ID)

    assert sorted(os.listdir(service._work)) == [other_job.name]


def test_files_in_the_working_folder_carry_the_job_tag(service, monkeypatch):
    seen = {}

    class Recording(WorkingYDL):
        def __init__(self, opts):
            super().__init__(opts)
            seen["outtmpl"] = opts["outtmpl"]

    monkeypatch.setattr(service.yt_dlp, "YoutubeDL", Recording)
    service.download_youtube_video_with_progress("https://example.com/v", job_id=JOB_ID)
    assert os.path.basename(seen["outtmpl"]).startswith(f"{TAG}_")


def test_the_download_only_file_carries_the_job_tag(service, monkeypatch):
    monkeypatch.setattr(service.yt_dlp, "YoutubeDL", WorkingYDL)
    path, _ = service.download_youtube_video_with_progress(
        "https://example.com/v", job_id=JOB_ID
    )
    assert os.path.basename(path) == f"Clip_{TAG}.mp4"
    assert os.listdir(service._work) == []


def test_the_pipeline_source_keeps_its_plain_title(service, monkeypatch):
    """The processing step shows this file's name as the video title; its own
    outputs already carry the processing task's id."""
    monkeypatch.setattr(service.yt_dlp, "YoutubeDL", WorkingYDL)
    path, _ = service.download_youtube_video("https://example.com/v", job_id=JOB_ID)
    assert os.path.basename(path) == "Clip.mp4"


def test_the_quality_choice_reaches_yt_dlp(service, monkeypatch):
    seen = []

    class Recording(WorkingYDL):
        def __init__(self, opts):
            super().__init__(opts)
            seen.append(opts["format"])

    monkeypatch.setattr(service.yt_dlp, "YoutubeDL", Recording)
    service.download_youtube_video_with_progress("https://example.com/v")
    service.download_youtube_video_with_progress("https://example.com/v", "high")
    assert seen == [format_for_quality("fast"), format_for_quality("high")]


# --------------------------------------------------------------------------- daily sweep


def test_the_daily_sweep_keeps_a_running_download(monkeypatch, tmp_path):
    from tasks import cleanup_tasks

    work, downloads = tmp_path / "work", tmp_path / "downloads"
    work.mkdir()
    downloads.mkdir()
    running = work / f"{TAG}_Clip.f137.mp4.part"
    running.write_bytes(b"x" * 1024)
    abandoned = work / "12345678_Old.f137.mp4.part"
    abandoned.write_bytes(b"x" * 1024)
    seven_hours_ago = time.time() - 7 * 3600
    os.utime(abandoned, (seven_hours_ago, seven_hours_ago))

    monkeypatch.setattr(
        cleanup_tasks,
        "config",
        MagicMock(
            DOWNLOADS_FOLDER=str(downloads),
            FAST_WORK_DIR=str(work),
            FAST_WORK_MAX_AGE=6 * 3600,
        ),
    )
    result = cleanup_tasks.cleanup_old_files_task.run(days=14)

    assert running.exists()
    assert not abandoned.exists()
    assert result["fast_work_removed"] == 1
