"""
Clean Performance Monitor for SubsTranslator
===========================================

Monitors system performance with clean, actionable logging.

Features:
- Smart thresholds (no spam for normal operations)
- EMA (Exponential Moving Average) for download speeds
- System resource monitoring (when available)
- Performance alerts only when actionable
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

# Optional psutil import
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting."""

    # Download performance (MB/s)
    download_speed_warning: float = 2.0  # Warn if < 2 MB/s average
    download_speed_critical: float = 0.5  # Critical if < 0.5 MB/s average

    # System resources (%)
    memory_usage_warning: float = 85.0
    disk_usage_warning: float = 90.0

    # Processing performance
    transcription_rtf_warning: float = 1.0  # Warn if RTF > 1.0 (slower than real-time)
    ffmpeg_speed_warning: float = 2.0  # Warn if speed ratio < 2.0x

    # File operations (seconds)
    file_move_warning: float = 5.0  # Warn if file move > 5s

    # Translation costs (USD)
    translation_cost_info: float = 0.01  # Log cost if > $0.01
    translation_cost_warning: float = 0.10  # Warn if > $0.10


class DownloadSpeedTracker:
    """Tracks download speed with EMA to avoid spam."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha  # EMA smoothing factor
        self.ema_speed: float | None = None
        self.last_update = 0
        self.min_update_interval = 2.0  # Minimum 2s between updates

    def update(self, size_mb: float, duration_s: float) -> float | None:
        """Update with new measurement, returns EMA speed if should log."""
        current_time = time.time()

        # Skip too-frequent updates
        if current_time - self.last_update < self.min_update_interval:
            return None

        if duration_s <= 0:
            return None

        current_speed = size_mb / duration_s

        if self.ema_speed is None:
            self.ema_speed = current_speed
        else:
            self.ema_speed = (
                self.alpha * current_speed + (1 - self.alpha) * self.ema_speed
            )

        self.last_update = current_time
        return self.ema_speed

    def get_current_speed(self) -> float | None:
        """Get current EMA speed without updating."""
        return self.ema_speed


class CleanPerformanceMonitor:
    """Clean performance monitor with smart alerting."""

    def __init__(self):
        self.thresholds = PerformanceThresholds()
        self.download_tracker = DownloadSpeedTracker()
        self.last_system_check = 0
        self.system_check_interval = 30.0  # Check system resources every 30s

        if not PSUTIL_AVAILABLE:
            logger.debug(
                "Performance monitor: psutil not available, system metrics disabled"
            )

    def log_download_performance(
        self, size_bytes: float, duration_s: float, operation: str = "download"
    ) -> None:
        """Log download performance with smart alerting."""
        if duration_s <= 0:
            return

        size_mb = size_bytes / (1024 * 1024)
        current_speed = size_mb / duration_s

        # Update EMA tracker
        ema_speed = self.download_tracker.update(size_mb, duration_s)

        # Only log if EMA updated and we have meaningful data
        if ema_speed is None or size_mb < 0.1:  # Skip tiny files
            return

        # Determine log level based on performance
        if ema_speed < self.thresholds.download_speed_critical:
            logger.warning(
                f"🐌 Very slow download: {ema_speed:.1f}MB/s avg "
                f"(current: {current_speed:.1f}MB/s, size: {size_mb:.1f}MB)"
            )
        elif ema_speed < self.thresholds.download_speed_warning:
            logger.warning(
                f"⚠️ Slow download: {ema_speed:.1f}MB/s avg " f"(size: {size_mb:.1f}MB)"
            )
        else:
            # Only log good performance for large files or in debug mode
            if size_mb > 10 or logger.isEnabledFor(logging.DEBUG):
                logger.info(
                    f"📊 Download: {ema_speed:.1f}MB/s avg " f"(size: {size_mb:.1f}MB)"
                )

    def log_transcription_performance(
        self,
        audio_duration_s: float,
        process_duration_s: float,
        model: str,
        segments_count: int = 0,
    ) -> None:
        """Log transcription performance with RTF calculation."""
        if audio_duration_s <= 0 or process_duration_s <= 0:
            return

        rtf = process_duration_s / audio_duration_s
        speed_ratio = audio_duration_s / process_duration_s

        # Determine log level
        if rtf > self.thresholds.transcription_rtf_warning:
            logger.warning(
                f"🧠 Slow transcription: {speed_ratio:.1f}× speed "
                f"(RTF: {rtf:.2f}, model: {model}, segments: {segments_count})"
            )
        else:
            logger.info(
                f"🧠 Transcription: {speed_ratio:.1f}× speed "
                f"(model: {model}, segments: {segments_count})"
            )

    def log_ffmpeg_performance(
        self, input_duration_s: float, process_duration_s: float, operation: str
    ) -> None:
        """Log FFmpeg performance with speed ratio."""
        if input_duration_s <= 0 or process_duration_s <= 0:
            return

        speed_ratio = input_duration_s / process_duration_s

        if speed_ratio < self.thresholds.ffmpeg_speed_warning:
            logger.warning(
                f"🎬 Slow {operation}: {speed_ratio:.1f}× speed "
                f"({process_duration_s:.1f}s for {input_duration_s:.1f}s video)"
            )
        else:
            logger.info(f"🎬 {operation}: {speed_ratio:.1f}× speed")

    def log_translation_cost(
        self, cost_usd: float, tokens_total: int, service: str, model: str = ""
    ) -> None:
        """Log translation cost if significant."""
        if cost_usd >= self.thresholds.translation_cost_warning:
            logger.warning(
                f"💰 High translation cost: ${cost_usd:.4f} "
                f"({tokens_total} tokens, {service})"
            )
        elif cost_usd >= self.thresholds.translation_cost_info:
            logger.info(
                f"💰 Translation cost: ${cost_usd:.4f} "
                f"({tokens_total} tokens, {service})"
            )

    def log_file_operation(
        self,
        operation: str,
        size_bytes: float,
        duration_s: float,
        source: str = "",
        destination: str = "",
    ) -> None:
        """Log file operation performance."""
        if duration_s <= 0:
            return

        size_mb = size_bytes / (1024 * 1024)
        speed_mbps = size_mb / duration_s

        if duration_s > self.thresholds.file_move_warning:
            logger.warning(
                f"🚛 Slow {operation}: {duration_s:.1f}s "
                f"({speed_mbps:.1f}MB/s, {size_mb:.1f}MB)"
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🚛 {operation}: {duration_s:.2f}s ({speed_mbps:.1f}MB/s)")

    def check_system_resources(self) -> None:
        """Check system resources (throttled to avoid spam)."""
        current_time = time.time()
        if current_time - self.last_system_check < self.system_check_interval:
            return

        if not PSUTIL_AVAILABLE:
            return

        try:
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > self.thresholds.memory_usage_warning:
                logger.warning(
                    f"🚨 High memory usage: {memory.percent:.1f}% "
                    f"(available: {memory.available / (1024**3):.1f}GB)"
                )

            # Disk check (downloads folder)
            try:
                disk = psutil.disk_usage(config.DOWNLOADS_FOLDER)
                if disk.percent > self.thresholds.disk_usage_warning:
                    logger.warning(
                        f"🚨 High disk usage: {disk.percent:.1f}% "
                        f"(free: {disk.free / (1024**3):.1f}GB)"
                    )
            except:
                pass  # Skip if path doesn't exist

            self.last_system_check = current_time

        except Exception as e:
            logger.debug(f"System resource check failed: {e}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary."""
        summary = {
            "download_speed_ema": self.download_tracker.get_current_speed(),
            "thresholds": {
                "download_speed_warning": self.thresholds.download_speed_warning,
                "transcription_rtf_warning": self.thresholds.transcription_rtf_warning,
                "memory_usage_warning": self.thresholds.memory_usage_warning,
            },
        }

        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                summary["memory_usage_percent"] = round(memory.percent, 1)
                summary["memory_available_gb"] = round(memory.available / (1024**3), 1)
            except:
                pass

        return summary


# Global instance
performance_monitor = CleanPerformanceMonitor()


# Convenience functions for backward compatibility
def log_download_performance(
    file_size_mb: float, duration_s: float, operation: str = "download"
):
    """Convenience function for download performance logging (backward compatibility)."""
    # Convert MB to bytes for the new implementation
    size_bytes = file_size_mb * 1024 * 1024
    performance_monitor.log_download_performance(size_bytes, duration_s, operation)


def log_move_performance(
    file_size_mb: float,
    duration_s: float,
    source: str = "fast_work",
    destination: str = "downloads",
):
    """Convenience function for file move performance logging (backward compatibility)."""
    # Convert MB to bytes for the new implementation
    size_bytes = file_size_mb * 1024 * 1024
    performance_monitor.log_file_operation(
        "file_move", size_bytes, duration_s, source, destination
    )


def log_transcription_performance(
    audio_duration_s: float,
    process_duration_s: float,
    model: str,
    segments_count: int = 0,
):
    """Convenience function for transcription performance logging."""
    performance_monitor.log_transcription_performance(
        audio_duration_s, process_duration_s, model, segments_count
    )


def log_ffmpeg_performance(
    input_duration_s: float, process_duration_s: float, operation: str
):
    """Convenience function for FFmpeg performance logging."""
    performance_monitor.log_ffmpeg_performance(
        input_duration_s, process_duration_s, operation
    )


def check_system_resources():
    """Convenience function for system resource checking."""
    performance_monitor.check_system_resources()
