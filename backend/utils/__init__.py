"""
Backend utility functions
"""

import re
import shutil

from config import get_config
from core.exceptions import FfmpegNotInstalledError
from logging_config import get_logger

config = get_config()
logger = get_logger(__name__)


def check_ffmpeg():
    """
    Check if FFmpeg is installed and available in the system's PATH.

    Raises:
        FfmpegNotInstalledError: If FFmpeg is not found in system PATH

    Example:
        >>> check_ffmpeg()  # Passes if FFmpeg is installed
        >>> check_ffmpeg()  # Raises FfmpegNotInstalledError if not found
    """
    if not shutil.which("ffmpeg"):
        logger.error(
            "FFmpeg not found. Please install FFmpeg and ensure it is in your system's PATH."
        )
        raise FfmpegNotInstalledError(
            "FFmpeg is not installed. This is required for video processing."
        )


def allowed_file(filename):
    """
    Check if file extension is allowed based on configuration.

    Args:
        filename: Name of the file to check

    Returns:
        bool: True if file extension is allowed, False otherwise

    Example:
        >>> allowed_file("video.mp4")
        True
        >>> allowed_file("video.exe")
        False
    """
    return config.is_allowed_file_extension(filename)


def sanitize_filename(filename):
    """
    Sanitize a filename by removing or replacing unsafe characters.

    Args:
        filename: The filename to sanitize

    Returns:
        str: Sanitized filename safe for file system use

    Example:
        >>> sanitize_filename("my/file:name*.txt")
        'my_file_name_.txt'
    """
    # Replace unsafe characters with underscores
    unsafe_chars = r'[<>:"/\\|?*]'
    safe_filename = re.sub(unsafe_chars, "_", filename)

    # Remove leading/trailing dots and spaces
    safe_filename = safe_filename.strip(". ")

    # Limit filename length to 255 characters (common filesystem limit)
    if len(safe_filename) > 255:
        name, ext = (
            safe_filename.rsplit(".", 1)
            if "." in safe_filename
            else (safe_filename, "")
        )
        max_name_len = 255 - len(ext) - 1  # -1 for the dot
        safe_filename = f"{name[:max_name_len]}.{ext}" if ext else name[:255]

    return safe_filename
