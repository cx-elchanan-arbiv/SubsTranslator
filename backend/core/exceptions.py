"""
Structured exception hierarchy for SubsTranslator.
Provides consistent error handling with error codes, user messages, and recovery hints.
"""

# =============================================================================
# Simple Base Exceptions (for backwards compatibility)
# =============================================================================


class AppError(Exception):
    """Base class for all application-specific errors."""

    pass


class FfmpegNotInstalledError(AppError):
    """Raised when FFmpeg is not installed or not in PATH (legacy compatibility)."""

    pass


class InvalidFileError(AppError):
    """Raised when an invalid file is provided."""

    pass


class TokenError(AppError):
    """Raised when download token validation fails."""

    pass


# =============================================================================
# Structured Exceptions with Error Codes
# =============================================================================


class VideoProcessingError(Exception):
    """Base class for all video processing errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        recoverable: bool = True,
        user_message: str | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        self.user_message = user_message or message
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "code": self.error_code,  # Use 'code' to match frontend TaskError interface
            "message": self.message,
            "user_facing_message": self.user_message,
            "recoverable": self.recoverable,
        }


class FFmpegError(VideoProcessingError):
    """Base class for FFmpeg-related errors."""

    pass


class FFmpegTimeoutError(FFmpegError):
    """Raised when FFmpeg operations timeout."""

    def __init__(self, operation: str, timeout: int):
        message = f"FFmpeg {operation} timeout after {timeout} seconds"
        user_message = f"The {operation} operation took too long. Please try again or choose a smaller file."
        super().__init__(
            message=message,
            error_code="FFMPEG_TIMEOUT",
            recoverable=True,
            user_message=user_message,
        )
        self.operation = operation
        self.timeout = timeout


class FFmpegProcessError(FFmpegError):
    """Raised when FFmpeg process fails."""

    def __init__(self, operation: str, stderr: str):
        message = f"FFmpeg {operation} failed: {stderr}"
        user_message = (
            f"Error processing {operation}. The file may be corrupted or unsupported."
        )
        super().__init__(
            message=message,
            error_code="FFMPEG_PROCESS_ERROR",
            recoverable=True,
            user_message=user_message,
        )
        self.operation = operation
        self.stderr = stderr


class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg is not installed or not found."""

    def __init__(self):
        message = "FFmpeg is not installed or not found in PATH"
        user_message = (
            "FFmpeg is not installed on the system. Please install FFmpeg to continue."
        )
        super().__init__(
            message=message,
            error_code="FFMPEG_NOT_FOUND",
            recoverable=False,
            user_message=user_message,
        )


class YouTubeDownloadError(VideoProcessingError):
    """Base class for YouTube download errors."""

    pass


class YouTubeAccessError(YouTubeDownloadError):
    """Raised when YouTube video is not accessible."""

    def __init__(self, url: str, reason: str):
        message = f"Cannot access YouTube video {url}: {reason}"
        user_message = "Cannot access the video. It may be private, age-restricted, or unavailable."
        super().__init__(
            message=message,
            error_code="YOUTUBE_ACCESS_ERROR",
            recoverable=False,
            user_message=user_message,
        )
        self.url = url
        self.reason = reason


class YouTubeNetworkError(YouTubeDownloadError):
    """Raised when YouTube download fails due to network issues."""

    def __init__(self, url: str, error: str):
        message = f"Network error downloading {url}: {error}"
        user_message = "Network error while downloading the video. Please check your internet connection and try again."
        super().__init__(
            message=message,
            error_code="YOUTUBE_NETWORK_ERROR",
            recoverable=True,
            user_message=user_message,
        )
        self.url = url
        self.network_error = error


class YouTubeBotDetectionError(YouTubeDownloadError):
    """Raised when YouTube detects and blocks automated download (bot detection)."""

    def __init__(self, url: str):
        message = f"YouTube bot detection blocked download for {url}"
        user_message = "YouTube is blocking downloads from our server. Please download the video to your computer and upload it as a file."
        super().__init__(
            message=message,
            error_code="YOUTUBE_BOT_DETECTION",
            recoverable=False,  # Not recoverable by retry
            user_message=user_message,
        )
        self.url = url


class TranscriptionError(VideoProcessingError):
    """Base class for transcription errors."""

    pass


class WhisperModelError(TranscriptionError):
    """Raised when Whisper model loading or processing fails."""

    def __init__(self, model: str, error: str):
        message = f"Whisper model {model} error: {error}"
        user_message = f"Error with transcription model {model}. Please try a different model or try again."
        super().__init__(
            message=message,
            error_code="WHISPER_MODEL_ERROR",
            recoverable=True,
            user_message=user_message,
        )
        self.model = model
        self.whisper_error = error


class AudioExtractionError(TranscriptionError):
    """Raised when audio extraction from video fails."""

    def __init__(self, video_path: str, error: str):
        message = f"Audio extraction failed for {video_path}: {error}"
        user_message = "Cannot extract audio from the video. The file may be corrupted or in an unsupported format."
        super().__init__(
            message=message,
            error_code="AUDIO_EXTRACTION_ERROR",
            recoverable=False,
            user_message=user_message,
        )
        self.video_path = video_path
        self.extraction_error = error


class TranslationError(VideoProcessingError):
    """Base class for translation errors."""

    pass


class TranslationServiceError(TranslationError):
    """Raised when translation service fails."""

    def __init__(self, service: str, error: str):
        message = f"Translation service {service} error: {error}"
        user_message = f"The translation service {service} is currently unavailable. Please try again later."
        super().__init__(
            message=message,
            error_code="TRANSLATION_SERVICE_ERROR",
            recoverable=True,
            user_message=user_message,
        )
        self.service = service
        self.service_error = error


class TranslationQuotaError(TranslationError):
    """Raised when translation service quota is exceeded."""

    def __init__(self, service: str):
        message = f"Translation service {service} quota exceeded"
        user_message = f"The translation quota for {service} has been exceeded. Please try again later or use a different service."
        super().__init__(
            message=message,
            error_code="TRANSLATION_QUOTA_ERROR",
            recoverable=True,
            user_message=user_message,
        )
        self.service = service


class FileProcessingError(VideoProcessingError):
    """Base class for file processing errors."""

    pass


class FileNotFoundError(FileProcessingError):
    """Raised when required file is not found."""

    def __init__(self, file_path: str):
        message = f"File not found: {file_path}"
        user_message = (
            "The file was not found. It may have been deleted or the path is incorrect."
        )
        super().__init__(
            message=message,
            error_code="FILE_NOT_FOUND",
            recoverable=False,
            user_message=user_message,
        )
        self.file_path = file_path


class FilePermissionError(FileProcessingError):
    """Raised when file permission is denied."""

    def __init__(self, file_path: str, operation: str):
        message = f"Permission denied for {operation} on {file_path}"
        user_message = (
            f"Permission denied to {operation} the file. Please check file permissions."
        )
        super().__init__(
            message=message,
            error_code="FILE_PERMISSION_ERROR",
            recoverable=False,
            user_message=user_message,
        )
        self.file_path = file_path
        self.operation = operation


class FileSizeError(FileProcessingError):
    """Raised when file size exceeds limits."""

    def __init__(self, file_path: str, size: int, max_size: int):
        message = f"File {file_path} size {size} exceeds maximum {max_size}"
        user_message = f"The file is too large ({size/1024/1024:.1f}MB). Maximum allowed size is {max_size/1024/1024:.1f}MB."
        super().__init__(
            message=message,
            error_code="FILE_SIZE_ERROR",
            recoverable=False,
            user_message=user_message,
        )
        self.file_path = file_path
        self.size = size
        self.max_size = max_size


class ConfigurationError(VideoProcessingError):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, value: str, expected: str):
        message = f"Invalid configuration {config_key}={value}, expected {expected}"
        user_message = (
            f"System configuration {config_key} is invalid. Please check the settings."
        )
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            recoverable=False,
            user_message=user_message,
        )
        self.config_key = config_key
        self.value = value
        self.expected = expected


# Utility functions for error handling
def handle_subprocess_error(e: Exception, operation: str) -> VideoProcessingError:
    """Convert subprocess errors to structured exceptions."""
    import subprocess

    if isinstance(e, subprocess.TimeoutExpired):
        return FFmpegTimeoutError(operation, e.timeout)
    elif isinstance(e, subprocess.CalledProcessError):
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        return FFmpegProcessError(operation, stderr)
    else:
        return VideoProcessingError(
            message=f"Subprocess error in {operation}: {str(e)}",
            error_code="SUBPROCESS_ERROR",
            recoverable=True,
            user_message=f"Error processing {operation}. Please try again.",
        )


def handle_youtube_error(e: Exception, url: str) -> YouTubeDownloadError:
    """Convert yt-dlp errors to structured exceptions."""
    error_str = str(e).lower()

    # Check for bot detection first (most specific)
    if any(
        keyword in error_str
        for keyword in ["sign in to confirm", "confirm you're not a bot", "bot"]
    ):
        return YouTubeBotDetectionError(url)
    elif any(
        keyword in error_str
        for keyword in ["private", "unavailable", "404", "not found"]
    ):
        return YouTubeAccessError(url, str(e))
    elif any(keyword in error_str for keyword in ["network", "timeout", "connection"]):
        return YouTubeNetworkError(url, str(e))
    else:
        return YouTubeDownloadError(
            message=f"YouTube download error: {str(e)}",
            error_code="YOUTUBE_DOWNLOAD_ERROR",
            recoverable=True,
            user_message="Error downloading the video from YouTube. Please try again or choose a different video.",
        )
