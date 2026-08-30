"""
URL resolver service.

Probes any URL (a direct video link OR a webpage that *contains* video) without
downloading, using yt-dlp's generic extractor, and reports what was found:

  {"type": "single",   "video":  {...}}            # exactly one video
  {"type": "multiple", "videos": [{...}, ...]}      # several -> let the user pick
  {"type": "none",     "reason": "<key>"}           # no extractable video

"Several" means the URL itself is a collection: a page carrying more than one
embedded video, or a playlist. A URL that names ONE video is always "single" —
even when it also carries playlist context, which is what a YouTube link copied
mid-playlist looks like (``watch?v=X&list=RD...``).

This is the entry point for "paste a page URL, not just a direct video link".
It does NOT handle JS-rendered / signed-token pages (e.g. Maven) — those need a
headless browser and are explicitly out of scope here (see docs/URL_PAGE_EXTRACTION_POC.md).
"""

import yt_dlp

from config import get_config
from logging_config import get_logger

config = get_config()
logger = get_logger(__name__)

#: How many candidates the picker is allowed to show.
MAX_CANDIDATES = 50


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
        # True, not False. The old value came from a wrong belief: that
        # noplaylist=False was what let a multi-video PAGE reveal all its
        # entries. It is not. --no-playlist only disambiguates a URL that names
        # a video AND a playlist at once; a page with several embeds, or a real
        # playlist URL, still returns every entry. Measured on all four shapes.
        #
        # With False, a link copied while a YouTube mix was playing
        # (watch?v=X&list=RD...) expanded into the whole radio station: one real
        # case returned 467 videos, and the song the user actually pasted was
        # buried at position 135.
        "noplaylist": True,
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

        # Ordering depends on where the list came from, because the two sources
        # mean different things:
        #   generic page - arbitrary embeds, so longest first is a decent guess
        #                  at "the content" over a trailer or an ad clip
        #   playlist     - the order IS the information. Sorting a playlist by
        #                  duration scrambles it and hides what the user pointed at.
        if (info.get("extractor") or "").lower() == "generic":
            videos.sort(key=lambda v: v.get("duration") or 0, reverse=True)

        total = len(videos)
        result = {"type": "multiple", "videos": videos[:MAX_CANDIDATES]}
        if total > MAX_CANDIDATES:
            # A long playlist is a legitimate thing to paste; a picker with
            # hundreds of rows is not a way to choose from it.
            result["truncated"] = True
            result["total"] = total
        return result

    return {"type": "single", "video": _candidate(info, url)}
