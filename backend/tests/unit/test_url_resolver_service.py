"""
Unit tests for the URL resolver (page-URL -> video discovery) and the yt-dlp
format fallback that lets non-YouTube HLS sources (Fox News, TED) download.

Covers the logic added for "paste a webpage that contains video, not just a
direct link" — see docs/URL_PAGE_EXTRACTION_POC.md.
"""

import os
import sys

import pytest
import yt_dlp

backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from unittest.mock import MagicMock, patch

from config import get_config
from services.url_resolver_service import resolve_video_url


def _mock_ydl(monkeyless_return=None, side_effect=None):
    """Build a patched yt_dlp.YoutubeDL whose extract_info returns/raises."""
    patcher = patch("yt_dlp.YoutubeDL")
    mock_cls = patcher.start()
    mock_ydl = MagicMock()
    mock_cls.return_value.__enter__.return_value = mock_ydl
    if side_effect is not None:
        mock_ydl.extract_info.side_effect = side_effect
    else:
        mock_ydl.extract_info.return_value = monkeyless_return
    return patcher


@pytest.mark.unit
class TestResolveVideoUrl:
    """resolve_video_url classifies a URL as single / multiple / none."""

    def test_single_video(self):
        info = {
            "title": "A talk",
            "duration": 600,
            "webpage_url": "https://site/talk",
            "thumbnail": "https://site/t.jpg",
            "uploader": "Someone",
        }
        p = _mock_ydl(monkeyless_return=info)
        try:
            res = resolve_video_url("https://site/talk")
        finally:
            p.stop()
        assert res["type"] == "single"
        assert res["video"]["title"] == "A talk"
        assert res["video"]["duration"] == 600
        assert res["video"]["duration_string"] == "10:00"

    def test_generic_page_is_sorted_longest_first(self):
        """A page's embeds have no meaningful order, so longest-first is a
        reasonable guess at "the content" over a trailer or an ad clip."""
        info = {
            "extractor": "generic",
            "entries": [
                {"title": "Ad", "duration": 30, "url": "https://site/ad"},
                {"title": "Main talk", "duration": 754, "url": "https://site/main"},
                {"title": "Intro", "duration": 45, "url": "https://site/intro"},
            ],
        }
        p = _mock_ydl(monkeyless_return=info)
        try:
            res = resolve_video_url("https://site/page")
        finally:
            p.stop()
        assert res["type"] == "multiple"
        assert len(res["videos"]) == 3
        assert res["videos"][0]["title"] == "Main talk"
        assert res["videos"][0]["duration_string"] == "12:34"

    def test_playlist_order_is_left_alone(self):
        """A playlist's order IS the information. Sorting it by duration
        scrambles it — that is how a pasted song ended up at position 135 of a
        467-entry radio mix instead of being the thing that downloaded."""
        info = {
            "extractor": "youtube:tab",
            "entries": [
                {"title": "Track 1", "duration": 120, "url": "https://yt/1"},
                {"title": "Track 2", "duration": 900, "url": "https://yt/2"},
                {"title": "Track 3", "duration": 200, "url": "https://yt/3"},
            ],
        }
        p = _mock_ydl(monkeyless_return=info)
        try:
            res = resolve_video_url("https://youtube.com/playlist?list=X")
        finally:
            p.stop()
        assert [v["title"] for v in res["videos"]] == ["Track 1", "Track 2", "Track 3"]

    def test_no_playlist_is_requested_so_a_video_in_a_playlist_stays_one_video(self):
        """The regression itself, at the level where it was caused.

        ``watch?v=X&list=RD...`` — a link copied while a YouTube mix played —
        names one video. Asking yt-dlp with noplaylist=False expanded it into
        the entire radio station. The flag is the whole fix, so assert the flag.
        """
        captured = {}

        patcher = patch("yt_dlp.YoutubeDL")
        mock_cls = patcher.start()

        def record(opts):
            captured.update(opts)
            return mock_cls.return_value

        mock_cls.side_effect = record
        mock_ydl = MagicMock()
        mock_cls.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"title": "One song", "duration": 300}
        try:
            res = resolve_video_url(
                "https://www.youtube.com/watch?v=X&list=RDX&start_radio=1"
            )
        finally:
            patcher.stop()

        assert captured.get("noplaylist") is True
        assert res["type"] == "single"

    def test_a_huge_playlist_is_capped(self):
        """A picker with hundreds of rows is not a way to choose from a list."""
        from services.url_resolver_service import MAX_CANDIDATES

        count = MAX_CANDIDATES + 25
        info = {
            "extractor": "youtube:tab",
            "entries": [
                {"title": f"Track {i}", "duration": 100 + i, "url": f"https://yt/{i}"}
                for i in range(count)
            ],
        }
        p = _mock_ydl(monkeyless_return=info)
        try:
            res = resolve_video_url("https://youtube.com/playlist?list=big")
        finally:
            p.stop()
        assert len(res["videos"]) == MAX_CANDIDATES
        assert res["truncated"] is True
        assert res["total"] == count

    def test_a_short_list_is_not_marked_truncated(self):
        info = {
            "extractor": "generic",
            "entries": [
                {"title": "A", "duration": 10, "url": "a"},
                {"title": "B", "duration": 20, "url": "b"},
            ],
        }
        p = _mock_ydl(monkeyless_return=info)
        try:
            res = resolve_video_url("https://site/two")
        finally:
            p.stop()
        assert "truncated" not in res
        assert "total" not in res

    def test_single_entry_collapses_to_single(self):
        info = {"entries": [{"title": "Only one", "duration": 12, "url": "u"}]}
        p = _mock_ydl(monkeyless_return=info)
        try:
            res = resolve_video_url("https://site/one")
        finally:
            p.stop()
        assert res["type"] == "single"
        assert res["video"]["title"] == "Only one"

    def test_empty_entries_is_none(self):
        p = _mock_ydl(monkeyless_return={"entries": []})
        try:
            res = resolve_video_url("https://site/empty")
        finally:
            p.stop()
        assert res["type"] == "none"
        assert res["reason"] == "no_video"

    def test_unsupported_url_is_no_video(self):
        err = yt_dlp.utils.DownloadError("ERROR: Unsupported URL: https://site/x")
        p = _mock_ydl(side_effect=err)
        try:
            res = resolve_video_url("https://site/x")
        finally:
            p.stop()
        assert res["type"] == "none"
        assert res["reason"] == "no_video"

    def test_login_required_is_needs_login(self):
        err = yt_dlp.utils.DownloadError(
            "ERROR: [vimeo] 123: The web client only works when logged-in. Use --cookies"
        )
        p = _mock_ydl(side_effect=err)
        try:
            res = resolve_video_url("https://vimeo.com/123")
        finally:
            p.stop()
        assert res["type"] == "none"
        assert res["reason"] == "needs_login"

    def test_unexpected_error_is_unavailable(self):
        p = _mock_ydl(side_effect=ValueError("boom"))
        try:
            res = resolve_video_url("https://site/boom")
        finally:
            p.stop()
        assert res["type"] == "none"
        assert res["reason"] == "unavailable"


@pytest.mark.unit
class TestOptimizedFormatFallbacks:
    """The format string must keep the YouTube remux path AND have generic
    fallbacks so adaptive HLS sources (Fox News / TED) don't fail."""

    def test_generic_adaptive_precedes_single_progressive(self):
        # Prefer working HLS streams over a single progressive file that some
        # sites (TED's h264-1200k) serve but then reject with HTTP 403.
        fmt = get_config().YTDLP_OPTIMIZED_FORMAT
        assert fmt.index("bestvideo[height<=1080]+bestaudio") < fmt.index(
            "best[ext=mp4]"
        )
