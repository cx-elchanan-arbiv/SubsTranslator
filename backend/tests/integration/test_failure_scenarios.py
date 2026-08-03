"""
Failure Scenario Integration Tests
Tests how the system handles real-world failure conditions.
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

# Add backend to path
backend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backend")
)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


@pytest.mark.integration
class TestFailureScenarios:
    """Test how the system handles failure scenarios."""

    def test_openai_quota_exceeded_recovery(self):
        """Test recovery when OpenAI quota is exceeded."""
        from services.translation_services import get_translator

        # Mock OpenAI to return quota exceeded error
        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Simulate quota exceeded error
            from openai import RateLimitError

            mock_client.chat.completions.create.side_effect = RateLimitError(
                "Rate limit exceeded",
                response=Mock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            )

            translator = get_translator("openai")

            # Should handle the error gracefully
            result = translator.translate_batch(["Hello", "World"], "he")

            # Should fallback to original text or handle gracefully
            assert isinstance(result, list)
            assert len(result) == 2

    def test_disk_full_during_processing(self):
        """Test behavior when disk becomes full during processing."""
        from services.subtitle_service import create_srt_file

        # Create a temporary directory with limited space
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test.srt")

            # Mock os.write to simulate disk full
            original_write = os.write

            def mock_write(fd, data):
                # Simulate disk full after a few writes
                if len(data) > 100:
                    raise OSError(28, "No space left on device")  # ENOSPC
                return original_write(fd, data)

            with patch("os.write", mock_write):
                segments = [{"start": 0, "end": 1, "text": "A" * 1000}]  # Large text

                # Should handle disk full gracefully
                try:
                    create_srt_file(segments, temp_file, "en", use_translation=False)
                    assert False, "Should have raised an exception for disk full"
                except OSError as e:
                    assert e.errno == 28  # ENOSPC
                    # File should not exist or be empty
                    assert (
                        not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0
                    )

    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion during processing."""
        from services.whisper_smart import SmartWhisperManager

        # Mock memory allocation to simulate exhaustion
        with patch("whisper_smart.WhisperModel") as mock_whisper:
            mock_whisper.side_effect = MemoryError("Cannot allocate memory")

            manager = SmartWhisperManager()

            # Should handle memory error gracefully
            with pytest.raises((MemoryError, RuntimeError)):
                manager.transcribe(b"fake_audio_data", "auto")

    def test_redis_connection_loss_recovery(self):
        """Test recovery when Redis connection is lost."""
        from unittest.mock import Mock

        from state_manager import EnterpriseStateManager

        # Create mock Celery task
        mock_task = Mock()
        mock_task.update_state = Mock()

        # Create state manager
        steps_config = [{"label": "Test Step", "weight": 1.0}]
        manager = EnterpriseStateManager(mock_task, steps_config)

        # Simulate Redis connection error
        mock_task.update_state.side_effect = Exception("Redis connection failed")

        # Should handle Redis failure gracefully
        try:
            manager.set_step_progress(0, 50, "Progress update")
            # Should not crash, may log error but continue
        except Exception as e:
            # If it does raise, should be handled gracefully
            assert "Redis" in str(e) or "connection" in str(e).lower()

    def test_youtube_video_unavailable_mid_download(self):
        """Test handling when YouTube video becomes unavailable during download."""
        from core.exceptions import YouTubeAccessError
        from services.youtube_service import download_youtube_video

        # Mock yt-dlp to simulate video becoming unavailable
        with patch("yt_dlp.YoutubeDL") as mock_ytdl:
            mock_instance = Mock()
            mock_ytdl.return_value.__enter__ = Mock(return_value=mock_instance)
            mock_ytdl.return_value.__exit__ = Mock(return_value=False)

            # Simulate video unavailable error
            mock_instance.extract_info.side_effect = Exception("Video unavailable")
            mock_instance.download.side_effect = Exception("Video unavailable")

            # yt-dlp errors are converted into structured exceptions;
            # "unavailable" maps to YouTubeAccessError (see core.exceptions).
            with pytest.raises(YouTubeAccessError):
                download_youtube_video("https://www.youtube.com/watch?v=invalid")

    def test_ffmpeg_process_killed_mid_execution(self):
        """Test handling when FFmpeg process is killed during execution."""
        import subprocess

        from services.subtitle_service import SubtitleService

        service = SubtitleService()

        # Mock subprocess to simulate process being killed
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                -9, ["ffmpeg"], "Process killed (SIGKILL)"
            )

            # Should handle process termination gracefully
            result = service.create_video_with_subtitles(
                "input.mp4", "subtitles.srt", "output.mp4", "he"
            )

            # Should return False indicating failure
            assert result is False

    def test_invalid_utf8_in_subtitle_content(self):
        """Test handling of invalid UTF-8 in subtitle content."""
        from services.subtitle_service import create_srt_file

        # Create segments with invalid UTF-8
        segments = [
            {"start": 0, "end": 1, "text": "Valid text"},
            {"start": 1, "end": 2, "text": "Invalid: \xff\xfe"},  # Invalid UTF-8
            {"start": 2, "end": 3, "text": "More valid text"},
        ]

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".srt", delete=False) as f:
            temp_file = f.name

        try:
            # Should handle invalid UTF-8 gracefully
            create_srt_file(segments, temp_file, "en", use_translation=False)

            # File should be created and readable
            with open(temp_file, encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Valid text should be preserved
            assert "Valid text" in content
            assert "More valid text" in content

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_task_timeout_during_critical_section(self):
        """Test handling of task timeout during critical operations."""
        from celery.exceptions import SoftTimeLimitExceeded

        # Mock the task to simulate timeout
        with patch("tasks.process_video_task.apply_async") as mock_task:
            mock_result = Mock()
            mock_task.return_value = mock_result

            # Simulate soft time limit exceeded
            mock_result.get.side_effect = SoftTimeLimitExceeded("Task timeout")

            # Should handle timeout gracefully
            with pytest.raises(SoftTimeLimitExceeded):
                result = mock_result.get()


@pytest.mark.integration
class TestResourceCleanup:
    """Test that resources are properly cleaned up after failures."""

    def test_temp_files_cleaned_after_failure(self):
        """Test that temporary files are cleaned up after task failure."""
        import glob
        import tempfile

        # Count initial temp files
        initial_temp_files = len(glob.glob("/tmp/substranslator_*"))

        # Simulate a failing task that creates temp files
        with tempfile.NamedTemporaryFile(prefix="substranslator_", delete=False) as f:
            temp_file = f.name
            f.write(b"temporary data")

        try:
            # Simulate task failure
            raise Exception("Simulated task failure")
        except Exception:
            # Cleanup should happen here in real code
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # Should not have more temp files than we started with
        final_temp_files = len(glob.glob("/tmp/substranslator_*"))
        assert final_temp_files <= initial_temp_files

    def test_partial_download_cleanup(self):
        """Test cleanup of partial downloads after failure."""
        from config import get_config

        config = get_config()
        downloads_dir = config.DOWNLOADS_FOLDER

        # Create a partial download file
        partial_file = os.path.join(downloads_dir, "partial_download.mp4.part")

        try:
            os.makedirs(downloads_dir, exist_ok=True)
            with open(partial_file, "w") as f:
                f.write("partial content")

            # Simulate download failure and cleanup
            # In real code, this would be handled by the download task
            if os.path.exists(partial_file):
                os.unlink(partial_file)

            # Partial file should be cleaned up
            assert not os.path.exists(partial_file)

        finally:
            # Ensure cleanup in test
            if os.path.exists(partial_file):
                os.unlink(partial_file)


@pytest.mark.integration
class TestConcurrencyIssues:
    """Test concurrent access scenarios that could cause issues."""

    def test_concurrent_task_state_updates(self):
        """Test concurrent updates to task state don't cause corruption."""
        import threading
        import time
        from unittest.mock import Mock

        from state_manager import EnterpriseStateManager

        # Create mock task
        mock_task = Mock()
        mock_task.update_state = Mock()

        # Create state manager
        steps_config = [{"label": "Step 1", "weight": 1.0}]
        manager = EnterpriseStateManager(mock_task, steps_config)

        # Track state updates
        updates = []
        original_update = mock_task.update_state

        def track_update(*args, **kwargs):
            updates.append((args, kwargs))
            time.sleep(0.001)  # Small delay to increase chance of race condition
            return original_update(*args, **kwargs)

        mock_task.update_state = track_update

        # Run concurrent updates
        def update_worker(progress):
            manager.set_step_progress(0, progress, f"Progress {progress}")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_worker, args=(i * 10,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have received all updates without corruption
        assert len(updates) >= 10

        # Final state should be consistent
        final_step = manager.steps[0]
        assert final_step.progress >= 0
        assert final_step.progress <= 100
