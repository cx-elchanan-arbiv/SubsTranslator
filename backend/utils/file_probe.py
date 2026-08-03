#!/usr/bin/env python3
"""
File Metadata Extraction using ffprobe
Extracts metadata from local video/audio files for unified UX display.
"""

import json
import os
import subprocess
from typing import Any

from logging_config import get_logger

logger = get_logger(__name__)


class FileProbeError(Exception):
    """Base exception for file probing errors"""

    pass


class UnsupportedMediaError(FileProbeError):
    """Raised when file is not a supported media format"""

    pass


class ProbeFailedError(FileProbeError):
    """Raised when ffprobe fails to extract metadata"""

    pass


def format_duration_string(duration_seconds: float) -> str:
    """
    Format duration in seconds to human-readable string (H:MM:SS or M:SS)

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted string like "3:45" or "1:23:45"
    """
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


def extract_file_metadata(file_path: str) -> dict[str, Any]:
    """
    Extract metadata from a local media file using ffprobe.

    Args:
        file_path: Absolute path to the media file

    Returns:
        Dictionary with file metadata:
        {
            "filename": "example.mp4",
            "file_size_mb": 15.3,
            "size_bytes": 16049152,
            "duration": 180.5,
            "duration_string": "3:01",
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "mime_type": "video/mp4",
            "extension": ".mp4",
            "codec_name": "h264",
            "audio_codec": "aac",
            "bit_rate": 2500000,
            "thumbnail_url": null
        }

    Raises:
        FileNotFoundError: If file doesn't exist
        UnsupportedMediaError: If file is not a valid media file
        ProbeFailedError: If ffprobe fails for other reasons
    """
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Path is not a file: {file_path}")

    # Get basic file info
    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()
    size_bytes = os.path.getsize(file_path)
    size_mb = round(size_bytes / (1024 * 1024), 2)

    # Build ffprobe command
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]

    try:
        logger.debug(f"Running ffprobe on: {filename}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )

        probe_data = json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timeout for file: {filename}")
        raise ProbeFailedError("File analysis timed out (30s)")

    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {filename}: {e.stderr}")
        raise UnsupportedMediaError(
            "File is not a valid media file or format is not supported"
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
        raise ProbeFailedError("Failed to parse media file information")

    # Extract format info
    format_info = probe_data.get("format", {})
    duration = float(format_info.get("duration", 0))
    bit_rate = int(format_info.get("bit_rate", 0))

    # Find video stream
    video_stream = None
    audio_stream = None

    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video" and not video_stream:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and not audio_stream:
            audio_stream = stream

    # If no video stream found, it might be audio-only or unsupported
    if not video_stream and not audio_stream:
        raise UnsupportedMediaError("No video or audio streams found in file")

    # Extract video info (if available)
    width = 0
    height = 0
    fps = 0.0
    codec_name = "unknown"

    if video_stream:
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec_name = video_stream.get("codec_name", "unknown")

        # Calculate FPS
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            if int(den) > 0:
                fps = round(int(num) / int(den), 2)
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    # Extract audio codec
    audio_codec = "none"
    if audio_stream:
        audio_codec = audio_stream.get("codec_name", "unknown")

    # Determine MIME type based on extension
    mime_map = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
    }
    mime_type = mime_map.get(extension, "application/octet-stream")

    # Build metadata dictionary
    metadata = {
        "filename": filename,
        "file_size_mb": size_mb,
        "size_bytes": size_bytes,
        "duration": round(duration, 2),
        "duration_string": format_duration_string(duration),
        "width": width,
        "height": height,
        "fps": fps,
        "mime_type": mime_type,
        "extension": extension,
        "codec_name": codec_name,
        "audio_codec": audio_codec,
        "bit_rate": bit_rate,
        "thumbnail_url": None,  # Could generate thumbnail in future
    }

    logger.info(
        f"📊 Extracted metadata for {filename}: "
        f"{width}x{height}, {format_duration_string(duration)}, {size_mb}MB"
    )

    return metadata


def probe_file_safe(file_path: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Safe wrapper around extract_file_metadata that returns (metadata, error_code).

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (metadata_dict, error_code):
        - On success: (metadata, None)
        - On error: (None, error_code)

    Error codes:
        - "FILE_NOT_FOUND": File doesn't exist
        - "UNSUPPORTED_MEDIA": File is not a valid media file
        - "PROBE_FAILED": ffprobe failed for other reasons
    """
    try:
        metadata = extract_file_metadata(file_path)
        return metadata, None

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None, "FILE_NOT_FOUND"

    except UnsupportedMediaError as e:
        logger.warning(f"Unsupported media: {e}")
        return None, "UNSUPPORTED_MEDIA"

    except ProbeFailedError as e:
        logger.error(f"Probe failed: {e}")
        return None, "PROBE_FAILED"

    except Exception as e:
        logger.exception(f"Unexpected error during file probe: {e}")
        return None, "PROBE_FAILED"
