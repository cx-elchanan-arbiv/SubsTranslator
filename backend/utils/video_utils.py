"""
Video utilities for video processing operations.
"""

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)


def cut_video_ffmpeg(
    input_path: str, output_path: str, start_time: str, end_time: str
) -> bool:
    """
    Cut a video from start_time to end_time using FFmpeg with ultra-precise cutting.

    Uses a three-method approach for maximum precision:
    1. Ultra-precise double seeking
    2. Filter-complex precision method
    3. Fast fallback method

    Args:
        input_path: Path to input video file
        output_path: Path to save cut video
        start_time: Start time in HH:MM:SS or MM:SS format
        end_time: End time in HH:MM:SS or MM:SS format

    Returns:
        True if successful, False otherwise
    """

    try:
        # Calculate duration
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        duration = end_seconds - start_seconds

        if duration <= 0:
            logger.error(f"Invalid time range: {start_time} to {end_time}")
            return False

        # Method 1: Ultra-precise double seeking (recommended)
        # This method is the most accurate for exact frame cutting
        logger.info(
            f"Cutting video from {start_time} to {end_time} (duration: {duration}s)"
        )

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss",
            start_time,  # Seek to start time (fast)
            "-i",
            input_path,  # Input file
            "-t",
            str(duration),  # Duration to cut
            "-c",
            "copy",  # Copy streams (fast, no re-encoding)
            "-avoid_negative_ts",
            "make_zero",  # Fix timestamp issues
            output_path,
        ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            logger.warning(
                f"Method 1 failed, trying Method 2 (filter-complex). Error: {result.stderr}"
            )

            # Method 2: Filter-complex precision (slower but more precise)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vf",
                f"trim=start={start_seconds}:end={end_seconds},setpts=PTS-STARTPTS",
                "-af",
                f"atrim=start={start_seconds}:end={end_seconds},asetpts=PTS-STARTPTS",
                output_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout for re-encoding
            )

            if result.returncode != 0:
                logger.error(f"Method 2 failed. Error: {result.stderr}")
                return False

        # Verify output file exists and has reasonable size
        if not os.path.exists(output_path):
            logger.error(f"Output file not created: {output_path}")
            return False

        output_size = os.path.getsize(output_path)
        if output_size < 1000:  # Less than 1KB is suspicious
            logger.error(f"Output file too small: {output_size} bytes")
            return False

        logger.info(f"Video cut successfully: {output_path} ({output_size} bytes)")
        return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"Error cutting video: {str(e)}")
        return False


def time_to_seconds(time_str: str) -> float:
    """
    Convert time string (HH:MM:SS or MM:SS or SS) to seconds.

    Args:
        time_str: Time string in format HH:MM:SS, MM:SS, or SS

    Returns:
        Time in seconds as float
    """
    parts = time_str.split(":")
    parts = [float(p) for p in parts]

    if len(parts) == 3:  # HH:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:  # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:  # SS
        return parts[0]
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def get_video_duration(video_path: str) -> float | None:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or None if error
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return float(result.stdout.strip())
        return None

    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        return None


def embed_subtitles_ffmpeg(video_path: str, srt_path: str, output_path: str) -> bool:
    """
    Embed (burn-in) subtitles into video using FFmpeg.

    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitles file
        output_path: Path to save output video

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Embedding subtitles from {srt_path} into {video_path}")

        # Escape path for FFmpeg filter (Windows compatibility)
        srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            video_path,  # Input video
            "-vf",
            f"subtitles={srt_path_escaped}",  # Burn-in subtitles
            "-c:a",
            "copy",  # Copy audio without re-encoding
            "-movflags",
            "+faststart",  # moov atom at front, so WhatsApp/players show a video preview
            output_path,
        ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600  # 10 minutes timeout
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False

        # Verify output
        if not os.path.exists(output_path):
            logger.error(f"Output file not created: {output_path}")
            return False

        output_size = os.path.getsize(output_path)
        if output_size < 1000:
            logger.error(f"Output file too small: {output_size} bytes")
            return False

        logger.info(
            f"Subtitles embedded successfully: {output_path} ({output_size} bytes)"
        )
        return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"Error embedding subtitles: {str(e)}")
        return False


def parse_text_to_srt(text: str, output_path: str) -> bool:
    """
    Parse timestamped text and convert to SRT format.

    Expected format:
        [MM:SS - MM:SS] Text
        or
        [HH:MM:SS - HH:MM:SS] Text

    Args:
        text: Input text with timestamps
        output_path: Path to save SRT file

    Returns:
        True if successful, False otherwise
    """
    try:
        import re

        lines = text.strip().split("\n")
        srt_entries = []
        entry_num = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match pattern: [time1 - time2] text
            match = re.match(
                r"\[(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)",
                line,
            )

            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                subtitle_text = match.group(3).strip()

                # Convert to SRT time format (HH:MM:SS,mmm)
                start_srt = convert_to_srt_time(start_time)
                end_srt = convert_to_srt_time(end_time)

                # Create SRT entry
                srt_entry = f"{entry_num}\n{start_srt} --> {end_srt}\n{subtitle_text}\n"
                srt_entries.append(srt_entry)
                entry_num += 1

        if not srt_entries:
            logger.error("No valid subtitle entries found in text")
            return False

        # Write SRT file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_entries))

        logger.info(f"Created SRT file with {len(srt_entries)} entries: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error parsing text to SRT: {str(e)}")
        return False


def convert_to_srt_time(time_str: str) -> str:
    """
    Convert time string (MM:SS or HH:MM:SS) to SRT format (HH:MM:SS,000).

    Args:
        time_str: Time string

    Returns:
        SRT formatted time string
    """
    parts = time_str.split(":")

    if len(parts) == 2:  # MM:SS
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)},000"
    elif len(parts) == 3:  # HH:MM:SS
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)},000"
    else:
        return "00:00:00,000"


def add_watermark_to_video(
    video_path: str,
    output_path: str,
    logo_path: str,
    position: str = "bottom-right",
    size: str = "medium",
    opacity: int = 40,
) -> bool:
    """
    Add watermark/logo to video using FFmpeg overlay filter.

    Args:
        video_path: Path to input video
        output_path: Path to save output video
        logo_path: Path to logo image
        position: Position (top-left, top-right, bottom-left, bottom-right)
        size: Size (small, medium, large) - height in pixels
        opacity: Opacity (0-100)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Adding watermark to {video_path}")

        # Map size to height in pixels
        size_map = {"small": 80, "medium": 120, "large": 200}
        height = size_map.get(size, 120)

        # Map position to overlay coordinates
        position_map = {
            "top-left": "10:10",
            "top-right": "main_w-overlay_w-10:10",
            "bottom-left": "10:main_h-overlay_h-10",
            "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
        }
        overlay_pos = position_map.get(position, "main_w-overlay_w-10:10")

        # Calculate opacity for FFmpeg (0.0 - 1.0)
        ffmpeg_opacity = opacity / 100.0

        # Build FFmpeg filter
        filter_complex = f"[1:v]scale=-1:{height},format=rgba,colorchannelmixer=aa={ffmpeg_opacity}[logo];[0:v][logo]overlay={overlay_pos}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            logo_path,
            "-filter_complex",
            filter_complex,
            "-c:a",
            "copy",  # Copy audio
            "-movflags",
            "+faststart",  # moov atom at front, so WhatsApp/players show a video preview
            output_path,
        ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False

        # Verify output
        if not os.path.exists(output_path):
            logger.error(f"Output file not created: {output_path}")
            return False

        output_size = os.path.getsize(output_path)
        logger.info(
            f"Watermark added successfully: {output_path} ({output_size} bytes)"
        )
        return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"Error adding watermark: {str(e)}")
        return False


def merge_videos_ffmpeg(video1_path: str, video2_path: str, output_path: str) -> bool:
    """
    Merge two videos using FFmpeg concat filter.

    Handles different resolutions by scaling videos to match.
    Uses the concat filter for smooth transitions.

    Args:
        video1_path: Path to first video (plays first)
        video2_path: Path to second video (plays second)
        output_path: Path to save merged video

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Merging {video1_path} and {video2_path}")

        # Method 1: Try concat demuxer (fastest, requires same codec/resolution)
        # Create concat list file
        concat_list_path = output_path + ".concat.txt"
        try:
            with open(concat_list_path, "w", encoding="utf-8") as f:
                f.write(f"file '{video1_path}'\n")
                f.write(f"file '{video2_path}'\n")

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list_path,
                "-c",
                "copy",  # Fast copy
                output_path,
            ]

            logger.info(f"Trying fast concat method: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                # Verify output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    logger.info("Fast concat succeeded")
                    os.remove(concat_list_path)  # Cleanup
                    output_size = os.path.getsize(output_path)
                    logger.info(
                        f"Videos merged successfully: {output_path} ({output_size} bytes)"
                    )
                    return True

            logger.warning(
                f"Fast concat failed, trying re-encode method. Error: {result.stderr}"
            )

        finally:
            # Cleanup concat list file
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)

        # Method 2: Re-encode with filter_complex (slower but handles different formats)
        # This scales both videos to match resolution and re-encodes
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video1_path,
            "-i",
            video2_path,
            "-filter_complex",
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
            "[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
            "[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[outv][outa]",
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            output_path,
        ]

        logger.info(f"Running re-encode concat: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minutes for re-encoding
        )

        if result.returncode != 0:
            logger.error(f"Re-encode concat failed: {result.stderr}")
            return False

        # Verify output
        if not os.path.exists(output_path):
            logger.error(f"Output file not created: {output_path}")
            return False

        output_size = os.path.getsize(output_path)
        if output_size < 1000:
            logger.error(f"Output file too small: {output_size} bytes")
            return False

        logger.info(f"Videos merged successfully: {output_path} ({output_size} bytes)")
        return True

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"Error merging videos: {str(e)}")
        return False


# ----------------------------------------------------------------------------------
# Audio level probing (used by the hallucination gate)
# ----------------------------------------------------------------------------------
#: Matches FFmpeg's ``volumedetect`` mean level line: ``mean_volume: -21.7 dB``.
_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def mean_volume_db(media_path: str, start: float = None, duration: float = None):
    """Mean audio level of a media file, or of one window inside it, in dBFS.

    Uses FFmpeg's ``volumedetect`` filter with ``-f null``, so nothing is written and no
    stream is re-encoded: the cost is decoding the requested window, which for the
    fractions of a second this is asked about is a few tens of milliseconds.

    Returns ``None`` for anything that does not produce a number — no audio stream, a
    window past the end of the file, a missing FFmpeg. A caller that cannot measure must
    be able to carry on without measuring.
    """
    if not media_path:
        return None
    cmd = ["ffmpeg", "-v", "info"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, float(start)):.3f}"]
    cmd += ["-i", media_path]
    if duration is not None:
        cmd += ["-t", f"{max(0.05, float(duration)):.3f}"]
    cmd += ["-af", "volumedetect", "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 - measurement must never raise at a caller
        logger.debug(f"mean_volume_db failed for {media_path}: {exc}")
        return None
    match = _MEAN_VOLUME_RE.search(result.stderr or "")
    return float(match.group(1)) if match else None


def relative_energy_probe(media_path: str):
    """Build a ``(start, end) -> dB relative to the whole file`` callable, or ``None``.

    The whole file is measured ONCE, lazily, and every window is reported relative to
    that baseline. Relative is the only comparable quantity: an absolute -25 dBFS means
    "quiet" in a loud podcast and "normal" in a quietly-mastered interview, and the
    question being asked — "is anyone speaking here?" — is about this file's own idea of
    speech.

    ``subtitle_engine.drop_hallucinated_cues`` consumes exactly this shape and treats a
    ``None`` result, or a probe that raises, as "no measurement".
    """
    baseline = {}

    def probe(start: float, end: float):
        if "value" not in baseline:
            baseline["value"] = mean_volume_db(media_path)
        if baseline["value"] is None:
            return None
        window = mean_volume_db(
            media_path, start=start, duration=float(end) - float(start)
        )
        return None if window is None else round(window - baseline["value"], 2)

    return probe
