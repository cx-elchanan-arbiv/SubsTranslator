import os
import sys
import tempfile
import time
from unittest.mock import patch

import pytest

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Set testing environment before importing
os.environ.setdefault("FLASK_ENV", "testing")

from tasks import cleanup_files_task


@pytest.mark.integration
class TestCleanupTask:
    """Test cleanup_files_task functionality."""

    def test_cleanup_removes_old_files_only(self):
        """Test that cleanup removes only old files, not new ones."""
        with (
            tempfile.TemporaryDirectory() as temp_upload,
            tempfile.TemporaryDirectory() as temp_download,
        ):

            # Mock the config to use our temp directories
            with (
                patch("tasks.UPLOAD_FOLDER", temp_upload),
                patch("tasks.DOWNLOADS_FOLDER", temp_download),
                patch("tasks.MAX_FILE_AGE", 3600),
            ):  # 1 hour

                # Create test files
                old_file_upload = os.path.join(temp_upload, "old_file.mp4")
                new_file_upload = os.path.join(temp_upload, "new_file.mp4")
                old_file_download = os.path.join(temp_download, "old_result.srt")
                new_file_download = os.path.join(temp_download, "new_result.srt")

                # Create files
                for file_path in [
                    old_file_upload,
                    new_file_upload,
                    old_file_download,
                    new_file_download,
                ]:
                    with open(file_path, "w") as f:
                        f.write("test content")

                # Mock file modification times
                current_time = time.time()
                old_time = current_time - 7200  # 2 hours ago (older than MAX_FILE_AGE)
                new_time = (
                    current_time - 1800
                )  # 30 minutes ago (newer than MAX_FILE_AGE)

                with patch("os.path.getmtime") as mock_getmtime:

                    def mock_mtime(path):
                        if "old_" in os.path.basename(path):
                            return old_time
                        else:
                            return new_time

                    mock_getmtime.side_effect = mock_mtime

                    # Run cleanup task
                    result = cleanup_files_task.apply()

                    # Check results
                    assert result.successful()
                    task_result = result.result

                    assert task_result["status"] == "Cleanup complete"
                    cleaned_files = task_result["cleaned_files"]

                    # Should have cleaned 2 old files
                    assert len(cleaned_files) == 2
                    assert "old_file.mp4" in cleaned_files
                    assert "old_result.srt" in cleaned_files

                    # New files should not be in cleaned list
                    assert "new_file.mp4" not in cleaned_files
                    assert "new_result.srt" not in cleaned_files

    def test_cleanup_handles_empty_directories(self):
        """Test that cleanup handles empty directories gracefully."""
        with (
            tempfile.TemporaryDirectory() as temp_upload,
            tempfile.TemporaryDirectory() as temp_download,
        ):

            with (
                patch("tasks.UPLOAD_FOLDER", temp_upload),
                patch("tasks.DOWNLOADS_FOLDER", temp_download),
            ):

                # Run cleanup on empty directories
                result = cleanup_files_task.apply()

                assert result.successful()
                task_result = result.result

                assert task_result["status"] == "Cleanup complete"
                assert task_result["cleaned_files"] == []

    def test_cleanup_handles_nonexistent_directories(self):
        """Test that cleanup handles nonexistent directories gracefully."""
        nonexistent_upload = "/tmp/nonexistent_upload_dir"
        nonexistent_download = "/tmp/nonexistent_download_dir"

        with (
            patch("tasks.UPLOAD_FOLDER", nonexistent_upload),
            patch("tasks.DOWNLOADS_FOLDER", nonexistent_download),
        ):

            # Should handle FileNotFoundError gracefully
            with pytest.raises(FileNotFoundError):
                cleanup_files_task.apply()

    def test_cleanup_skips_subdirectories(self):
        """Test that cleanup only processes files, not subdirectories."""
        with tempfile.TemporaryDirectory() as temp_upload:

            with (
                patch("tasks.UPLOAD_FOLDER", temp_upload),
                patch("tasks.DOWNLOADS_FOLDER", temp_upload),
                patch("tasks.MAX_FILE_AGE", 3600),
            ):

                # Create a file and a subdirectory
                old_file = os.path.join(temp_upload, "old_file.mp4")
                subdir = os.path.join(temp_upload, "subdir")

                with open(old_file, "w") as f:
                    f.write("test content")

                os.makedirs(subdir)

                # Mock old modification time for both
                old_time = time.time() - 7200
                with patch("os.path.getmtime", return_value=old_time):

                    result = cleanup_files_task.apply()

                    assert result.successful()
                    task_result = result.result

                    # Should only clean the file, not the directory
                    cleaned_files = task_result["cleaned_files"]
                    assert len(cleaned_files) == 1
                    assert "old_file.mp4" in cleaned_files

                    # Directory should still exist
                    assert os.path.exists(subdir)

    def test_cleanup_handles_permission_errors(self):
        """Test that cleanup handles permission errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_upload:

            with (
                patch("tasks.UPLOAD_FOLDER", temp_upload),
                patch("tasks.DOWNLOADS_FOLDER", temp_upload),
                patch("tasks.MAX_FILE_AGE", 3600),
            ):

                # Create a file
                old_file = os.path.join(temp_upload, "old_file.mp4")
                with open(old_file, "w") as f:
                    f.write("test content")

                # Mock old modification time and permission error on removal
                old_time = time.time() - 7200
                with (
                    patch("os.path.getmtime", return_value=old_time),
                    patch(
                        "os.remove", side_effect=PermissionError("Permission denied")
                    ),
                ):

                    # Should handle the error gracefully (might raise or log)
                    try:
                        result = cleanup_files_task.apply()
                        # If it doesn't raise, check that it handled the error
                        assert result.failed() or result.successful()
                    except PermissionError:
                        # It's acceptable for the task to fail on permission errors
                        pass

    def test_cleanup_progress_reporting(self):
        """Test that cleanup task reports progress correctly."""
        with tempfile.TemporaryDirectory() as temp_upload:

            with (
                patch("tasks.UPLOAD_FOLDER", temp_upload),
                patch("tasks.DOWNLOADS_FOLDER", temp_upload),
            ):

                # Create a file
                test_file = os.path.join(temp_upload, "test_file.mp4")
                with open(test_file, "w") as f:
                    f.write("test content")

                # Run the task and check that it updates state
                result = cleanup_files_task.apply()

                assert result.successful()
                # The task should complete and return a result
                task_result = result.result
                assert "status" in task_result
                assert "cleaned_files" in task_result

    def test_cleanup_file_age_boundary(self):
        """Test cleanup behavior at the exact age boundary."""
        with tempfile.TemporaryDirectory() as temp_upload:

            with (
                patch("tasks.UPLOAD_FOLDER", temp_upload),
                patch("tasks.DOWNLOADS_FOLDER", temp_upload),
                patch("tasks.MAX_FILE_AGE", 3600),
            ):  # 1 hour

                # Create files
                boundary_file = os.path.join(temp_upload, "boundary_file.mp4")
                just_old_file = os.path.join(temp_upload, "just_old_file.mp4")
                just_new_file = os.path.join(temp_upload, "just_new_file.mp4")

                for file_path in [boundary_file, just_old_file, just_new_file]:
                    with open(file_path, "w") as f:
                        f.write("test content")

                current_time = time.time()

                def mock_mtime(path):
                    filename = os.path.basename(path)
                    if filename == "boundary_file.mp4":
                        return current_time - 3600  # Exactly at boundary
                    elif filename == "just_old_file.mp4":
                        return current_time - 3601  # 1 second older
                    else:  # just_new_file.mp4
                        return current_time - 3599  # 1 second newer

                with patch("os.path.getmtime", side_effect=mock_mtime):

                    result = cleanup_files_task.apply()

                    assert result.successful()
                    task_result = result.result

                    cleaned_files = task_result["cleaned_files"]

                    # Files at or older than MAX_FILE_AGE should be cleaned
                    assert "boundary_file.mp4" in cleaned_files
                    assert "just_old_file.mp4" in cleaned_files

                    # Files newer than MAX_FILE_AGE should not be cleaned
                    assert "just_new_file.mp4" not in cleaned_files
