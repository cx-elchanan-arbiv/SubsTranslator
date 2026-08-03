"""
URL resolver service.

Probes any URL (a direct video link OR a webpage that *contains* video) without
downloading, using yt-dlp's generic extractor, and reports what was found:

  {"type": "single",   "video":  {...}}            # exactly one video
  {"type": "multiple", "videos": [{...}, ...]}      # several -> let the user pick
  {"type": "none",     "reason": "<key>"}           # no extractable video

This is the entry point for "paste a page URL, not just a direct video link".
It does NOT handle JS-rendered / signed-token pages (e.g. Maven) — those need a
headless browser and are explicitly out of scope here (see docs/URL_PAGE_EXTRACTION_POC.md).
"""

import yt_dlp

from config import get_config
from logging_config import get_logger

config = get_config()
logger = get_logger(__name__)


def _duration_string(seconds: float | None) -> str:
    """Format seconds as H:MM:SS / M:SS (mirrors yt-dlp's duration_string)."""
    if not seconds:
        return ""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _candidate(entry: dict, fallback_url: str) -> dict:
    """Normalize a yt-dlp info/entry dict into a lightweight candidate."""
    duration = entry.get("duration")
    return {
        "url": entry.get("webpage_url") or entry.get("url") or fallback_url,
        "title": entry.get("title") or "Untitled",
        "duration": duration,
        "duration_string": entry.get("duration_string") or _duration_string(duration),
        "thumbnail": entry.get("thumbnail") or "",
        "uploader": entry.get("uploader") or "",
    }


def resolve_video_url(url: str) -> dict:
    """
    Probe `url` and classify what video(s) it exposes. Never downloads.

    Returns a dict with a "type" of "single" | "multiple" | "none".
    On "none", "reason" is one of: "no_video", "needs_login", "unavailable".
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": False,  # let multi-video pages reveal all entries
        "extract_flat": "in_playlist",  # fast: don't deep-fetch every entry
        "socket_timeout": config.YTDLP_SOCKET_TIMEOUT,
        "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if (
            "log in" in msg
            or "logged-in" in msg
            or "cookies" in msg
            or "account" in msg
        ):
            reason = "needs_login"
        elif (
            "unsupported url" in msg or "unable to extract" in msg or "no video" in msg
        ):
            reason = "no_video"
        else:
            reason = "unavailable"
        logger.info(f"resolve_video_url: no video for {url} ({reason})")
        return {"type": "none", "reason": reason, "detail": str(e)[:300]}
    except Exception as e:  # noqa: BLE001 - surface any unexpected failure as "none"
        logger.warning(f"resolve_video_url: unexpected error for {url}: {e}")
        return {"type": "none", "reason": "unavailable", "detail": str(e)[:300]}

    entries = info.get("entries")
    if entries is not None:
        # Filter out empty/None entries that flat extraction sometimes yields.
        videos = [_candidate(e, url) for e in entries if e]
        if len(videos) == 0:
            return {"type": "none", "reason": "no_video"}
        if len(videos) == 1:
            return {"type": "single", "video": videos[0]}
        # Longest first — the main content usually outlasts intros/ads.
        videos.sort(key=lambda v: v.get("duration") or 0, reverse=True)
        return {"type": "multiple", "videos": videos}

    return {"type": "single", "video": _candidate(info, url)}
