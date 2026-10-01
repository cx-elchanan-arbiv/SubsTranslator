"""
Download quality — the "fast (720p) / high quality (1080p)" choice in the UI.

Why it exists: downloads from video sites reach this machine at ~0.35-0.45 MB/s
(measured 2026-10-01 against CNN and YouTube, while the line itself does 2-3 MB/s),
so the size of the file IS the waiting time. A 720p file is ~40% smaller than the
1080p one (CNN, 44 s clip: 28 MB vs 45 MB; YouTube, 10.5 min: 353 MB vs 575 MB).

"fast" is the default, at the owner's request. The cost is a less sharp picture,
including in a video rendered with burned-in subtitles; transcription only uses the
audio and is unaffected.

Before this, every caller passed a ``quality`` string ("high", "medium") that nothing
read — the format was always capped at 1080p. The values still in flight map as
"high" -> 1080p and anything else -> the default.
"""

from config import get_config

config = get_config()

#: Highest video height each choice downloads.
QUALITY_MAX_HEIGHT = {"fast": 720, "high": 1080}

DEFAULT_QUALITY = "fast"

#: The cap written into config.YTDLP_OPTIMIZED_FORMAT; replaced per choice.
_FORMAT_CAP = "height<=1080"


def parse_quality(value) -> str:
    """The request/task value, as one of QUALITY_MAX_HEIGHT's keys."""
    if isinstance(value, str) and value.strip().lower() in QUALITY_MAX_HEIGHT:
        return value.strip().lower()
    return DEFAULT_QUALITY


def format_for_quality(quality) -> str:
    """
    The yt-dlp format string for ``quality``.

    Same preference order as config.YTDLP_OPTIMIZED_FORMAT (remux-friendly avc1/mp4a
    first, then generic adaptive, then a single file); only the height cap changes.
    The uncapped fallbacks at the end stay uncapped on purpose: they exist for sites
    whose formats carry no height, where a cap would mean "format not available".
    """
    height = QUALITY_MAX_HEIGHT[parse_quality(quality)]
    return config.YTDLP_OPTIMIZED_FORMAT.replace(_FORMAT_CAP, f"height<={height}")
