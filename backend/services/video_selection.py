"""
Getting exactly ONE video out of a URL that may hold several.

A page URL (a news live-blog, an article with embeds) is a playlist to yt-dlp. The
picker in the UI lets the user choose one entry, but some extractors give every entry
the PAGE's own URL as its ``webpage_url`` — CNN does, for all ten videos of a live
page — so the URL alone cannot say which one was chosen. The entry's ``id`` can, and
is what the resolver hands the picker as ``item_id`` in that case.

Without this, the page URL reached yt-dlp as a playlist: every video on it was
downloaded one after another (``noplaylist`` does not stop that — it only applies to
a URL naming a video AND a list), the progress bar sat at 92% from the second video
on, and the job ended on the task time limit or on a file named after the playlist
that does not exist. Measured on 2026-10-01: CNN live page, 10 videos, 5 downloaded
(430 MB) in 30 minutes, then SoftTimeLimitExceeded.

  item_filter_opts(item_id) - yt-dlp options that skip every entry but the chosen one
  single_video(info, ...)   - the one video's info dict out of what yt-dlp returned;
                              raises BEFORE anything is downloaded when there is not
                              exactly one
  parse_item_id(value)      - the request field, validated
"""

from core.exceptions import (
    PageHasMultipleVideosError,
    VideoNotOnPageError,
    YouTubeAccessError,
)

#: An item id travels from the browser, so it is bounded. Real ids are short
#: (CNN: 42 chars, YouTube: 11).
MAX_ITEM_ID_LENGTH = 200


def parse_item_id(value) -> str | None:
    """The ``item_id`` request field: a short non-empty string, or None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_ITEM_ID_LENGTH:
        return None
    return value


def item_filter_opts(item_id: str | None) -> dict:
    """
    yt-dlp options that keep only the entry whose id is ``item_id``.

    A callable rather than a ``"id = '...'"`` filter string, so an id is compared as
    a value and never parsed as filter syntax. Empty when nothing was chosen, so the
    options of an ordinary single-video URL are exactly what they were.
    """
    if not item_id:
        return {}

    def only_the_chosen_video(info, *, incomplete=False):
        video_id = info.get("id")
        if video_id is None and incomplete:
            return None  # a flat entry without an id yet: decided once it resolves
        if video_id == item_id:
            return None
        return "not the video chosen in the picker"

    return {"match_filter": only_the_chosen_video}


def single_video(info: dict, url: str, item_id: str | None = None) -> dict:
    """
    The info dict of the one video ``url`` stands for.

    ``info`` is what ``extract_info(url, download=False)`` returned: a video, or a
    playlist whose ``entries`` are what survived ``item_filter_opts``.
    """
    entries = info.get("entries")
    if entries is None:
        if item_id and info.get("id") != item_id:
            # The page now resolves straight to a different video.
            raise VideoNotOnPageError(url, item_id)
        return info

    videos = [entry for entry in entries if entry]
    if item_id:
        videos = [video for video in videos if video.get("id") == item_id]
        if not videos:
            raise VideoNotOnPageError(url, item_id)
    if len(videos) == 1:
        return videos[0]
    if not videos:
        # An empty page. The extractors usually raise for this themselves.
        raise YouTubeAccessError(url, "the page holds no video")
    raise PageHasMultipleVideosError(url, len(videos))
