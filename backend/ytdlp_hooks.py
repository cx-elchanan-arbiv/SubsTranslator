"""
Clean yt-dlp Progress Hooks
==========================

Provides clean, consolidated progress reporting for yt-dlp downloads.
Reduces spam and provides meaningful progress updates.

Features:
- Consolidated progress (no spam)
- EMA speed calculation
- Clean progress messages
- Integration with phase_logger
"""

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class CleanProgressTracker:
    """Clean progress tracker for yt-dlp downloads."""

    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self.last_update_time = 0
        self.start_time = time.time()

        # Progress tracking
        self.total_bytes: float | None = None
        self.downloaded_bytes = 0
        self.last_downloaded = 0

        # Speed tracking (EMA)
        self.speed_ema: float | None = None
        self.speed_alpha = 0.3  # EMA smoothing factor

        # State
        self.last_percent = 0
        self.is_downloading = False

    def should_update(self) -> bool:
        """Check if we should log an update."""
        current_time = time.time()
        return current_time - self.last_update_time >= self.update_interval

    def update_speed(self, current_time: float) -> None:
        """Update speed calculation with EMA."""
        if self.last_update_time == 0:
            self.last_update_time = current_time
            return

        time_diff = current_time - self.last_update_time
        if time_diff <= 0:
            return

        bytes_diff = self.downloaded_bytes - self.last_downloaded
        if bytes_diff <= 0:
            return

        # Calculate current speed in MB/s
        current_speed = (bytes_diff / (1024 * 1024)) / time_diff

        # Update EMA
        if self.speed_ema is None:
            self.speed_ema = current_speed
        else:
            self.speed_ema = (
                self.speed_alpha * current_speed
                + (1 - self.speed_alpha) * self.speed_ema
            )

    def format_size(self, bytes_val: float | None) -> str:
        """Format bytes as human readable size."""
        if bytes_val is None:
            return "?MB"

        mb = bytes_val / (1024 * 1024)
        if mb < 1:
            return f"{bytes_val / 1024:.1f}KB"
        elif mb < 1024:
            return f"{mb:.1f}MB"
        else:
            return f"{mb / 1024:.1f}GB"

    def log_progress(self, percent: float) -> None:
        """Log clean progress update."""
        current_time = time.time()

        # Update speed calculation
        self.update_speed(current_time)

        # Create progress message
        size_str = self.format_size(self.total_bytes)
        downloaded_str = self.format_size(self.downloaded_bytes)

        speed_str = ""
        if self.speed_ema and self.speed_ema > 0:
            speed_str = f" at {self.speed_ema:.1f}MB/s"

            # ETA calculation
            if self.total_bytes and self.speed_ema > 0:
                remaining_bytes = self.total_bytes - self.downloaded_bytes
                remaining_mb = remaining_bytes / (1024 * 1024)
                eta_seconds = remaining_mb / self.speed_ema

                if eta_seconds < 60:
                    eta_str = f" ETA {eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f" ETA {eta_seconds/60:.0f}m"
                else:
                    eta_str = f" ETA {eta_seconds/3600:.1f}h"
                speed_str += eta_str

        message = f"📥 Download: {percent:.1f}% of {size_str}{speed_str}"
        logger.info(message)

        # Update tracking
        self.last_update_time = current_time
        self.last_downloaded = self.downloaded_bytes
        self.last_percent = percent


def create_clean_progress_hook(
    phase_logger=None, update_interval: float = 2.0
) -> Callable:
    """
    Create a clean progress hook for yt-dlp.

    Args:
        phase_logger: Optional phase logger for integration
        update_interval: Minimum seconds between progress updates

    Returns:
        Progress hook function for yt-dlp
    """
    tracker = CleanProgressTracker(update_interval)

    def progress_hook(d: dict[str, Any]) -> None:
        """Clean yt-dlp progress hook."""
        try:
            status = d.get("status", "")

            if status == "downloading":
                tracker.is_downloading = True

                # Update tracking data
                if "total_bytes" in d and d["total_bytes"]:
                    tracker.total_bytes = float(d["total_bytes"])
                elif "total_bytes_estimate" in d and d["total_bytes_estimate"]:
                    tracker.total_bytes = float(d["total_bytes_estimate"])

                if "downloaded_bytes" in d:
                    tracker.downloaded_bytes = float(d["downloaded_bytes"])

                # Calculate percentage
                if tracker.total_bytes and tracker.total_bytes > 0:
                    percent = (tracker.downloaded_bytes / tracker.total_bytes) * 100
                    percent = min(100, max(0, percent))

                    # Only log if significant change or time passed
                    percent_diff = abs(percent - tracker.last_percent)
                    if percent_diff >= 10 or tracker.should_update():
                        tracker.log_progress(percent)

            elif status == "finished":
                if tracker.is_downloading:
                    # Log completion
                    total_time = time.time() - tracker.start_time
                    size_str = tracker.format_size(tracker.total_bytes)

                    avg_speed = ""
                    if tracker.total_bytes and total_time > 0:
                        avg_speed_mbps = (
                            tracker.total_bytes / (1024 * 1024)
                        ) / total_time
                        avg_speed = f" avg {avg_speed_mbps:.1f}MB/s"

                    filename = d.get("filename", "file")
                    if "/" in filename:
                        filename = filename.split("/")[-1]

                    logger.info(
                        f"📥 Download complete: {size_str} in {total_time:.1f}s{avg_speed}"
                    )

                    # Integration with phase logger
                    if phase_logger and hasattr(phase_logger, "complete_phase"):
                        phase_logger.complete_phase(
                            "download",
                            size_bytes=tracker.total_bytes,
                            output_path=filename,
                            note="downloaded",
                        )

            elif status == "error":
                error_msg = d.get("error", "Unknown error")
                logger.error(f"📥 Download error: {error_msg}")

                if phase_logger and hasattr(phase_logger, "log_error"):
                    phase_logger.log_error("download", str(error_msg))

        except Exception as e:
            # Don't let hook errors break the download
            logger.debug(f"Progress hook error: {e}")

    return progress_hook


def create_ytdlp_options(phase_logger=None, quality: str = "high") -> dict[str, Any]:
    """
    Create clean yt-dlp options with progress hook.

    Args:
        phase_logger: Optional phase logger for integration
        quality: Download quality preference

    Returns:
        yt-dlp options dictionary
    """
    from config import get_config

    config = get_config()

    # Quality format mapping
    format_map = {
        "low": "worst",
        "medium": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "high": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "best": "bestvideo+bestaudio/best",
    }

    options = {
        # Clean output - no progress spam
        "quiet": False,  # We want some output
        "no_warnings": False,
        "noprogress": True,  # Disable built-in progress
        # Our clean progress hook
        "progress_hooks": [create_clean_progress_hook(phase_logger)],
        # Format selection
        "format": format_map.get(quality, format_map["high"]),
        "merge_output_format": "mp4",
        # android_vr client so YouTube returns full-res formats (not SABR 360p)
        "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
        # Performance settings
        "socket_timeout": config.YTDLP_SOCKET_TIMEOUT,
        "retries": config.YTDLP_RETRIES,
        "fragment_retries": config.YTDLP_FRAGMENT_RETRIES,
        # File handling
        "restrictfilenames": config.YTDLP_RESTRICT_FILENAMES,
        "continuedl": config.YTDLP_CONTINUE_DL,
        # Cache
        "cachedir": config.YTDLP_CACHE_DIR,
    }

    return options
