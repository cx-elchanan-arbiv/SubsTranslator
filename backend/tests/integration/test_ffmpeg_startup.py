import os
import sys
from unittest.mock import patch

import pytest

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Set testing environment before importing app
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("TESTING", "true")

from core.exceptions import FfmpegNotInstalledError
from utils import check_ffmpeg


@pytest.mark.integration
class TestFFmpegStartup:
    """Test FFmpeg startup checks."""

    def test_ffmpeg_available(self):
        """Test that FFmpeg check passes when FFmpeg is available."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"  # Simulate FFmpeg found

            # Should not raise any exception
            try:
                check_ffmpeg()
            except Exception as e:
                pytest.fail(f"check_ffmpeg() raised {e} unexpectedly!")

    def test_ffmpeg_not_available(self):
        """Test that FFmpeg check fails when FFmpeg is not available."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Simulate FFmpeg not found

            with pytest.raises(FfmpegNotInstalledError):
                check_ffmpeg()

    def test_ffmpeg_error_message_hebrew_friendly(self):
        """Test that FFmpeg error messages are Hebrew-friendly."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Simulate FFmpeg not found

            try:
                check_ffmpeg()
            except FfmpegNotInstalledError as e:
                # Check that error message is informative
                assert "FFmpeg" in str(e)
                assert "required" in str(e)


@pytest.mark.unit
class TestFFmpegErrorHandling:
    """Test FFmpeg error handling."""

    def test_multiple_ffmpeg_checks(self):
        """Test multiple consecutive FFmpeg checks."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"

            # Multiple calls should work fine
            check_ffmpeg()
            check_ffmpeg()
            check_ffmpeg()

    def test_ffmpeg_check_with_different_paths(self):
        """Test FFmpeg check with different system paths."""
        paths = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]

        for path in paths:
            with patch("shutil.which") as mock_which:
                mock_which.return_value = path

                # Should work with any valid path
                try:
                    check_ffmpeg()
                except Exception as e:
                    pytest.fail(f"check_ffmpeg() failed with path {path}: {e}")
