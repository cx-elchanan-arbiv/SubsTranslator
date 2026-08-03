"""
FFmpeg helper utilities for integration tests.

Creates real (but tiny) video files for testing.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def make_video(
    path: str,
    color: str = "black",
    seconds: int = 1,
    audio: bool = False,
    width: int = 320,
    height: int = 180,
) -> bool:
    """
    Create a dummy video file using FFmpeg.

    Args:
        path: Output video file path
        color: Video color (black, red, blue, green, etc.)
        seconds: Video duration in seconds
        audio: Whether to include audio track
        width: Video width in pixels
        height: Video height in pixels

    Returns:
        True if video was created successfully, False otherwise

    Example:
        >>> make_video('/tmp/test.mp4', color='red', seconds=2, audio=True)
        True
    """
    try:
        # Base command for video
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f",
            "lavfi",
            "-t",
            str(seconds),
            "-i",
            f"color=c={color}:s={width}x{height}:r=25",
        ]

        # Add audio if requested
        if audio:
            cmd += [
                "-f",
                "lavfi",
                "-t",
                str(seconds),
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-shortest",
            ]

        # Output settings - ultrafast for speed
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "35", path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False

        # Verify file was created
        if not os.path.exists(path):
            logger.error(f"Output file not created: {path}")
            return False

        file_size = os.path.getsize(path)
        logger.info(f"Created video: {path} ({file_size} bytes)")
        return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        return False


def make_srt_file(path: str, num_subtitles: int = 3) -> bool:
    """
    Create a dummy SRT subtitle file.

    Args:
        path: Output SRT file path
        num_subtitles: Number of subtitle entries to create

    Returns:
        True if file was created successfully

    Example:
        >>> make_srt_file('/tmp/test.srt', num_subtitles=5)
        True
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            for i in range(1, num_subtitles + 1):
                start_sec = (i - 1) * 2
                end_sec = i * 2
                f.write(f"{i}\n")
                f.write(f"00:00:{start_sec:02d},000 --> 00:00:{end_sec:02d},000\n")
                f.write(f"Test subtitle {i}\n")
                f.write("\n")

        logger.info(f"Created SRT file: {path} ({num_subtitles} entries)")
        return True

    except Exception as e:
        logger.error(f"Error creating SRT file: {str(e)}")
        return False


def make_logo_image(path: str, width: int = 100, height: int = 50) -> bool:
    """
    Create a dummy PNG logo image using FFmpeg.

    Args:
        path: Output PNG file path
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        True if image was created successfully

    Example:
        >>> make_logo_image('/tmp/logo.png', width=200, height=100)
        True
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s={width}x{height}:d=1",
            "-frames:v",
            "1",
            path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False

        logger.info(f"Created logo image: {path}")
        return True

    except Exception as e:
        logger.error(f"Error creating logo image: {str(e)}")
        return False
