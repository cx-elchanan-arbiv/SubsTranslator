"""
Tests for structured exception hierarchy.
"""

import os
import subprocess
import sys

import pytest

# Add backend to path and import exceptions
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, backend_dir)

# Import the exceptions file directly
import importlib.util

spec = importlib.util.spec_from_file_location(
    "exceptions", os.path.join(backend_dir, "core", "exceptions.py")
)
exceptions_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exceptions_module)

# Import all exception classes
VideoProcessingError = exceptions_module.VideoProcessingError
FFmpegTimeoutError = exceptions_module.FFmpegTimeoutError
FFmpegProcessError = exceptions_module.FFmpegProcessError
FFmpegNotFoundError = exceptions_module.FFmpegNotFoundError
YouTubeAccessError = exceptions_module.YouTubeAccessError
YouTubeNetworkError = exceptions_module.YouTubeNetworkError
YouTubeDownloadError = exceptions_module.YouTubeDownloadError
WhisperModelError = exceptions_module.WhisperModelError
AudioExtractionError = exceptions_module.AudioExtractionError
TranslationServiceError = exceptions_module.TranslationServiceError
TranslationQuotaError = exceptions_module.TranslationQuotaError
FileNotFoundError = exceptions_module.FileNotFoundError
FilePermissionError = exceptions_module.FilePermissionError
FileSizeError = exceptions_module.FileSizeError
ConfigurationError = exceptions_module.ConfigurationError
handle_subprocess_error = exceptions_module.handle_subprocess_error
handle_youtube_error = exceptions_module.handle_youtube_error


@pytest.mark.unit
class TestVideoProcessingError:
    """Test base VideoProcessingError class."""

    def test_error_to_dict(self):
        error = VideoProcessingError(
            message="Test error",
            error_code="TEST_ERROR",
            recoverable=False,
            user_message="שגיאת בדיקה",
        )

        expected = {
            "code": "TEST_ERROR",  # Uses 'code' to match frontend TaskError interface
            "message": "Test error",
            "user_facing_message": "שגיאת בדיקה",
            "recoverable": False,
        }

        assert error.to_dict() == expected

    def test_default_user_message(self):
        error = VideoProcessingError(message="Test error", error_code="TEST_ERROR")

        assert error.user_message == "Test error"
        assert error.recoverable == True  # Default


@pytest.mark.unit
class TestFFmpegErrors:
    """Test FFmpeg-related errors."""

    def test_ffmpeg_timeout_error(self):
        error = FFmpegTimeoutError("conversion", 30)

        assert error.operation == "conversion"
        assert error.timeout == 30
        assert error.error_code == "FFMPEG_TIMEOUT"
        assert error.recoverable == True
        assert "conversion" in error.message
        assert "30 seconds" in error.message
        assert "conversion" in error.user_message  # English user message

    def test_ffmpeg_process_error(self):
        error = FFmpegProcessError("probe", "Invalid format")

        assert error.operation == "probe"
        assert error.stderr == "Invalid format"
        assert error.error_code == "FFMPEG_PROCESS_ERROR"
        assert error.recoverable == True
        assert "probe failed" in error.message
        assert "Invalid format" in error.message

    def test_ffmpeg_not_found_error(self):
        error = FFmpegNotFoundError()

        assert error.error_code == "FFMPEG_NOT_FOUND"
        assert error.recoverable == False
        assert "FFmpeg" in error.message
        assert "FFmpeg" in error.user_message  # English user message


@pytest.mark.unit
class TestYouTubeErrors:
    """Test YouTube-related errors."""

    def test_youtube_access_error(self):
        url = "https://youtube.com/watch?v=test"
        error = YouTubeAccessError(url, "Private video")

        assert error.url == url
        assert error.reason == "Private video"
        assert error.error_code == "YOUTUBE_ACCESS_ERROR"
        assert error.recoverable == False
        assert url in error.message
        assert "Private video" in error.message

    def test_youtube_network_error(self):
        url = "https://youtube.com/watch?v=test"
        error = YouTubeNetworkError(url, "Connection timeout")

        assert error.url == url
        assert error.network_error == "Connection timeout"
        assert error.error_code == "YOUTUBE_NETWORK_ERROR"
        assert error.recoverable == True
        assert "Network error" in error.user_message  # English user message


@pytest.mark.unit
class TestTranscriptionErrors:
    """Test transcription-related errors."""

    def test_whisper_model_error(self):
        error = WhisperModelError("large-v3", "Model not found")

        assert error.model == "large-v3"
        assert error.whisper_error == "Model not found"
        assert error.error_code == "WHISPER_MODEL_ERROR"
        assert error.recoverable == True
        assert "large-v3" in error.message

    def test_audio_extraction_error(self):
        error = AudioExtractionError("/path/to/video.mp4", "No audio stream")

        assert error.video_path == "/path/to/video.mp4"
        assert error.extraction_error == "No audio stream"
        assert error.error_code == "AUDIO_EXTRACTION_ERROR"
        assert error.recoverable == False


@pytest.mark.unit
class TestTranslationErrors:
    """Test translation-related errors."""

    def test_translation_service_error(self):
        error = TranslationServiceError("google", "API key invalid")

        assert error.service == "google"
        assert error.service_error == "API key invalid"
        assert error.error_code == "TRANSLATION_SERVICE_ERROR"
        assert error.recoverable == True

    def test_translation_quota_error(self):
        error = TranslationQuotaError("openai")

        assert error.service == "openai"
        assert error.error_code == "TRANSLATION_QUOTA_ERROR"
        assert error.recoverable == True
        assert "quota" in error.user_message  # English user message


@pytest.mark.unit
class TestFileErrors:
    """Test file-related errors."""

    def test_file_not_found_error(self):
        error = FileNotFoundError("/path/to/missing.mp4")

        assert error.file_path == "/path/to/missing.mp4"
        assert error.error_code == "FILE_NOT_FOUND"
        assert error.recoverable == False

    def test_file_permission_error(self):
        error = FilePermissionError("/path/to/file.mp4", "write")

        assert error.file_path == "/path/to/file.mp4"
        assert error.operation == "write"
        assert error.error_code == "FILE_PERMISSION_ERROR"
        assert error.recoverable == False

    def test_file_size_error(self):
        error = FileSizeError("/path/to/large.mp4", 1000000000, 500000000)

        assert error.file_path == "/path/to/large.mp4"
        assert error.size == 1000000000
        assert error.max_size == 500000000
        assert error.error_code == "FILE_SIZE_ERROR"
        assert error.recoverable == False
        assert "953.7MB" in error.user_message  # 1000000000 / 1024 / 1024


@pytest.mark.unit
class TestConfigurationError:
    """Test configuration errors."""

    def test_configuration_error(self):
        error = ConfigurationError("WHISPER_MODEL", "invalid", "tiny|base|medium|large")

        assert error.config_key == "WHISPER_MODEL"
        assert error.value == "invalid"
        assert error.expected == "tiny|base|medium|large"
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.recoverable == False


@pytest.mark.unit
class TestErrorHandlers:
    """Test error handling utility functions."""

    def test_handle_subprocess_timeout(self):
        timeout_error = subprocess.TimeoutExpired("ffmpeg", 30)
        result = handle_subprocess_error(timeout_error, "conversion")

        assert isinstance(result, FFmpegTimeoutError)
        assert result.operation == "conversion"
        assert result.timeout == 30

    def test_handle_subprocess_process_error(self):
        process_error = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"Invalid codec"
        )
        result = handle_subprocess_error(process_error, "encoding")

        assert isinstance(result, FFmpegProcessError)
        assert result.operation == "encoding"
        assert result.stderr == "Invalid codec"

    def test_handle_subprocess_generic_error(self):
        generic_error = RuntimeError("Something went wrong")
        result = handle_subprocess_error(generic_error, "processing")

        assert isinstance(result, VideoProcessingError)
        assert result.error_code == "SUBPROCESS_ERROR"
        assert "processing" in result.message

    def test_handle_youtube_private_video(self):
        youtube_error = Exception("Private video")
        result = handle_youtube_error(youtube_error, "https://youtube.com/watch?v=test")

        assert isinstance(result, YouTubeAccessError)
        assert result.url == "https://youtube.com/watch?v=test"
        assert result.reason == "Private video"

    def test_handle_youtube_network_error(self):
        youtube_error = Exception("Network timeout")
        result = handle_youtube_error(youtube_error, "https://youtube.com/watch?v=test")

        assert isinstance(result, YouTubeNetworkError)
        assert result.url == "https://youtube.com/watch?v=test"
        assert result.network_error == "Network timeout"

    def test_handle_youtube_generic_error(self):
        youtube_error = Exception("Unknown error")
        result = handle_youtube_error(youtube_error, "https://youtube.com/watch?v=test")

        assert isinstance(result, YouTubeDownloadError)
        assert result.error_code == "YOUTUBE_DOWNLOAD_ERROR"
        assert "Unknown error" in result.message


@pytest.mark.unit
class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""
