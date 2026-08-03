"""
Video processing service for SubsTranslator
Handles video format verification and conversion operations
"""

import json
import os
import subprocess

from config import get_config
from core.exceptions import FFmpegProcessError, FFmpegTimeoutError
from logging_config import get_logger

# Configuration
config = get_config()
logger = get_logger(__name__)


def verify_and_convert_video_format(video_path, target_path=None):
    """
    Verify video format and convert to H.264/AAC if needed for QuickTime compatibility.
    Returns the path to the final compatible video.
    """
    try:
        # Probe video format
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            video_path,
        ]

        try:
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=config.FFPROBE_TIMEOUT,
            )
            streams = json.loads(result.stdout).get("streams", [])
        except subprocess.TimeoutExpired:
            raise FFmpegTimeoutError("probe", config.FFPROBE_TIMEOUT)
        except subprocess.CalledProcessError as e:
            raise FFmpegProcessError(
                "probe", e.stderr.decode() if e.stderr else "Unknown error"
            )
        except json.JSONDecodeError:
            raise FFmpegProcessError(
                "probe", "Invalid JSON output - video file may be corrupted"
            )

        video_stream = None
        audio_stream = None

        for stream in streams:
            if stream.get("codec_type") == "video":
                video_stream = stream
            elif stream.get("codec_type") == "audio":
                audio_stream = stream

        # Check if conversion is needed
        needs_conversion = False
        conversion_reason = []

        if video_stream:
            vcodec = video_stream.get("codec_name", "").lower()
            if vcodec not in ["h264", "avc"]:
                needs_conversion = True
                conversion_reason.append(f"video codec: {vcodec} → h264")

        if audio_stream:
            acodec = audio_stream.get("codec_name", "").lower()
            if acodec not in ["aac", "mp4a"]:
                needs_conversion = True
                conversion_reason.append(f"audio codec: {acodec} → aac")

        if not needs_conversion:
            logger.info(
                "Video format check passed",
                check_type="format_check",
                video_path=os.path.basename(video_path),
                format="H.264/AAC",
            )
            return video_path

        # Convert to H.264/AAC
        if not target_path:
            base_name = os.path.splitext(video_path)[0]
            target_path = f"{base_name}_compatible.mp4"

        logger.info(
            "Converting video format",
            operation="format_conversion_start",
            video_path=os.path.basename(video_path),
            conversion_reasons=conversion_reason,
        )

        convert_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-y",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",  # Optimize for web/QuickTime
            target_path,
        ]

        try:
            subprocess.run(
                convert_cmd,
                check=True,
                capture_output=True,
                timeout=config.FFMPEG_RUN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise FFmpegTimeoutError("conversion", config.FFMPEG_RUN_TIMEOUT)
        except subprocess.CalledProcessError as e:
            raise FFmpegProcessError(
                "conversion", e.stderr.decode() if e.stderr else "Unknown error"
            )

        # Remove original if conversion successful
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            os.remove(video_path)
            logger.info(
                f"✅ Converted to QuickTime-compatible format: {os.path.basename(target_path)}"
            )
            return target_path
        else:
            logger.error(
                f"❌ Conversion failed, keeping original: {os.path.basename(video_path)}"
            )
            return video_path

    except Exception as e:
        logger.error(f"❌ Error checking/converting video format: {e}")
        return video_path  # Return original on error
