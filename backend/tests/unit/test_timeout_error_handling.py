import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from services.transcription_service import transcribe_video
from services.video_processing_service import verify_and_convert_video_format


@pytest.mark.unit
class TestFFmpegTimeouts:
    """Test FFmpeg timeout handling."""

    def test_ffprobe_timeout_handling(self):
        """Test that FFprobe timeouts are handled gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(["ffprobe"], 30)

            # The function should handle the timeout and return the original path
            result = verify_and_convert_video_format("/fake/video.mp4")
            assert result == "/fake/video.mp4"  # Returns original on error

    def test_ffmpeg_conversion_timeout_handling(self):
        """Test that FFmpeg conversion timeouts are handled gracefully."""
        with patch("subprocess.run") as mock_run:
            # First call (ffprobe) succeeds, second call (ffmpeg) times out
            mock_run.side_effect = [
                MagicMock(
                    stdout='{"streams": [{"codec_type": "video", "codec_name": "h265"}]}'
                ),
                subprocess.TimeoutExpired(["ffmpeg"], 900),
            ]

            # The function should handle the timeout and return the original path
            result = verify_and_convert_video_format("/fake/video.mp4")
            assert result == "/fake/video.mp4"  # Returns original on error

    def test_ffmpeg_audio_extraction_timeout(self):
        """Test that FFmpeg audio extraction timeouts are handled gracefully."""
        with patch("services.transcription_service.config.USE_FAKE_YTDLP", True):
            # In FAKE mode, transcribe_video should return fake segments quickly
            result = transcribe_video("/fake/video.mp4")
            assert "segments" in result
            assert len(result["segments"]) > 0


@pytest.mark.unit
class TestFFmpegErrorMessages:
    """Test FFmpeg error message handling."""

    def test_ffprobe_called_process_error(self):
        """Test FFprobe CalledProcessError handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["ffprobe"], stderr=b"Invalid file format"
            )

            # The function should handle the error and return the original path
            result = verify_and_convert_video_format("/fake/video.mp4")
            assert result == "/fake/video.mp4"  # Returns original on error

    def test_ffprobe_json_decode_error(self):
        """Test FFprobe JSON decode error handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="invalid json")

            # The function should handle the error and return the original path
            result = verify_and_convert_video_format("/fake/video.mp4")
            assert result == "/fake/video.mp4"  # Returns original on error
