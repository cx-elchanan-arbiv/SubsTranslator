"""``cleanup_files_task`` — the periodic reaper for uploads/ and downloads/.

The folders and the age cutoff are read from module globals bound at import time in
``tasks/cleanup_tasks.py``, so that is where they must be patched. They used to live on
``tasks`` itself; patching the old location silently did nothing and the task ran against
the real configured folders.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from tasks import cleanup_files_task

CLEANUP = "tasks.cleanup_tasks"
MAX_AGE = 3600


@pytest.fixture
def folders():
    """Two temp folders wired in as the task's upload/download targets."""
    with (
        tempfile.TemporaryDirectory() as uploads,
        tempfile.TemporaryDirectory() as downloads,
    ):
        with (
            patch(f"{CLEANUP}.UPLOAD_FOLDER", uploads),
            patch(f"{CLEANUP}.DOWNLOADS_FOLDER", downloads),
            patch(f"{CLEANUP}.MAX_FILE_AGE", MAX_AGE),
        ):
            yield uploads, downloads


def _touch(folder, name):
    path = os.path.join(folder, name)
    with open(path, "w") as fh:
        fh.write("x")
    return path


def _run_with_ages(ages):
    """Run the task with a frozen clock and per-basename ages in seconds.

    Freezing ``now`` is what makes the age assertions mean anything: with a live clock
    the elapsed time inside the task is added to every age, so a file pinned exactly at
    the cutoff drifts past it and the boundary case passes for the wrong reason.
    """
    now = 1_000_000.0

    def fake_mtime(path):
        return now - ages[os.path.basename(path)]

    with (
        patch(f"{CLEANUP}.time.time", return_value=now),
        patch("os.path.getmtime", side_effect=fake_mtime),
    ):
        return cleanup_files_task.apply()


@pytest.mark.integration
class TestCleanupTask:
    def test_old_files_go_and_fresh_files_stay(self, folders):
        uploads, downloads = folders
        _touch(uploads, "stale.mp4")
        _touch(uploads, "fresh.mp4")
        _touch(downloads, "stale.srt")
        _touch(downloads, "fresh.srt")

        result = _run_with_ages(
            {
                "stale.mp4": 7200,
                "fresh.mp4": 1800,
                "stale.srt": 7200,
                "fresh.srt": 1800,
            }
        )

        assert result.successful()
        cleaned = result.result["cleaned_files"]
        assert sorted(cleaned) == ["stale.mp4", "stale.srt"]
        # ...and the survivors are still on disk, not merely absent from the report.
        assert os.path.exists(os.path.join(uploads, "fresh.mp4"))
        assert os.path.exists(os.path.join(downloads, "fresh.srt"))
        assert not os.path.exists(os.path.join(uploads, "stale.mp4"))

    def test_the_cutoff_is_strict(self, folders):
        """A file exactly at MAX_FILE_AGE is kept; one second older is removed."""
        uploads, _ = folders
        for name in ("at_cutoff.mp4", "past_cutoff.mp4", "inside_cutoff.mp4"):
            _touch(uploads, name)

        result = _run_with_ages(
            {
                "at_cutoff.mp4": MAX_AGE,
                "past_cutoff.mp4": MAX_AGE + 1,
                "inside_cutoff.mp4": MAX_AGE - 1,
            }
        )

        assert result.successful()
        assert result.result["cleaned_files"] == ["past_cutoff.mp4"]

    def test_subdirectories_are_left_alone(self, folders):
        """Only files are reaped — a stale-looking directory must survive."""
        uploads, _ = folders
        _touch(uploads, "stale.mp4")
        subdir = os.path.join(uploads, "subdir")
        os.makedirs(subdir)

        result = _run_with_ages({"stale.mp4": 7200, "subdir": 7200})

        assert result.successful()
        assert result.result["cleaned_files"] == ["stale.mp4"]
        assert os.path.isdir(subdir)

    def test_empty_folders_report_nothing_cleaned(self, folders):
        result = _run_with_ages({})

        assert result.successful()
        assert result.result == {"status": "Cleanup complete", "cleaned_files": []}

    def test_a_missing_folder_fails_the_task_rather_than_passing_silently(self):
        """Documented contract: a vanished folder is an error, not a no-op cleanup.

        A cleanup that reports "0 files removed" because the folder was not there is the
        failure mode worth catching — it looks identical to a healthy run in the logs.
        """
        with (
            patch(f"{CLEANUP}.UPLOAD_FOLDER", "/nonexistent/uploads"),
            patch(f"{CLEANUP}.DOWNLOADS_FOLDER", "/nonexistent/downloads"),
        ):
            result = cleanup_files_task.apply()

        assert result.failed()
        assert isinstance(result.result, FileNotFoundError)
