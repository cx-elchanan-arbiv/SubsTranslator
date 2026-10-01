"""Choosing ONE video from a page that holds several — end to end on the server side.

The case that broke (2026-10-01): a CNN live page with ten videos. yt-dlp gives every
entry the PAGE's URL as ``webpage_url``, so the picker sent the same URL whichever
video was chosen, and the download treated the page as a playlist: it fetched every
video in turn until the 30-minute task limit killed it, five videos and 430 MB in.

These tests pin the three pieces that close it:

* the resolver hands an ``item_id`` exactly when the URL alone cannot identify a video;
* the download resolves first and refuses — before any byte — when there is not
  exactly one video, and downloads only the chosen one when there is;
* the metadata step describes that one video, caches it per video (not per page),
  and lets the precise error code through.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.exceptions import (  # noqa: E402
    PageHasMultipleVideosError,
    VideoNotOnPageError,
    handle_youtube_error,
)
from services.video_selection import (  # noqa: E402
    MAX_ITEM_ID_LENGTH,
    item_filter_opts,
    parse_item_id,
    single_video,
)

pytestmark = pytest.mark.unit

PAGE = "https://edition.cnn.com/2026/09/30/world/live-news/some-live-page"


def _cnn_like_page(n=3):
    """The shape yt-dlp's CNN extractor returns: resolved entries, all on the page URL."""
    return {
        "_type": "playlist",
        "extractor": "CNN",
        "webpage_url": PAGE,
        "entries": [
            {
                "id": f"me{i:040d}",
                "title": f"Clip {i}",
                "duration": 30 + i,
                "webpage_url": PAGE,
                "url": f"https://clips-media.example/{i}.mp4",
            }
            for i in range(n)
        ],
    }


# --------------------------------------------------------------------------- helpers


class TestParseItemId:
    def test_a_real_id_passes(self):
        assert parse_item_id("me4200e6ef1a58") == "me4200e6ef1a58"

    def test_whitespace_is_trimmed(self):
        assert parse_item_id("  abc  ") == "abc"

    @pytest.mark.parametrize("value", [None, "", "   ", 42, ["a"], {"id": "a"}])
    def test_anything_else_is_none(self, value):
        assert parse_item_id(value) is None

    def test_an_oversized_value_is_refused(self):
        assert parse_item_id("x" * (MAX_ITEM_ID_LENGTH + 1)) is None


class TestItemFilterOpts:
    def test_no_choice_leaves_the_options_untouched(self):
        assert item_filter_opts(None) == {}
        assert item_filter_opts("") == {}

    def test_only_the_chosen_entry_passes(self):
        keep = item_filter_opts("b")["match_filter"]
        assert keep({"id": "b"}) is None
        assert keep({"id": "a"})  # a skip reason
        assert keep({"id": "a"}, incomplete=True)

    def test_a_flat_entry_without_an_id_is_decided_later(self):
        keep = item_filter_opts("b")["match_filter"]
        assert keep({"url": "https://x"}, incomplete=True) is None

    def test_an_id_with_quotes_is_a_value_not_filter_syntax(self):
        tricky = "a' | id != 'b"
        keep = item_filter_opts(tricky)["match_filter"]
        assert keep({"id": tricky}) is None
        assert keep({"id": "b"})


class TestSingleVideo:
    def test_a_plain_video_is_returned_as_is(self):
        info = {"id": "v", "title": "One"}
        assert single_video(info, "https://youtube.com/watch?v=v") is info

    def test_the_chosen_entry_is_returned(self):
        page = _cnn_like_page()
        chosen = page["entries"][1]
        assert single_video(page, PAGE, chosen["id"]) is chosen

    def test_a_page_of_one_is_that_video(self):
        page = _cnn_like_page(1)
        assert single_video(page, PAGE)["title"] == "Clip 0"

    def test_several_videos_and_no_choice_is_refused(self):
        with pytest.raises(PageHasMultipleVideosError) as caught:
            single_video(_cnn_like_page(10), PAGE)
        assert caught.value.error_code == "PAGE_HAS_MULTIPLE_VIDEOS"
        assert caught.value.count == 10
        assert caught.value.recoverable is False

    def test_a_chosen_video_gone_from_the_page_is_said_so(self):
        with pytest.raises(VideoNotOnPageError) as caught:
            single_video(_cnn_like_page(), PAGE, "me-removed-since")
        assert caught.value.error_code == "VIDEO_NOT_ON_PAGE"

    def test_a_page_that_now_resolves_to_another_video_is_said_so(self):
        with pytest.raises(VideoNotOnPageError):
            single_video({"id": "other", "title": "Other"}, PAGE, "chosen")

    def test_the_classified_error_survives_the_youtube_error_mapper(self):
        """handle_youtube_error guesses from message text; "is no longer on" must
        not be re-filed as a generic access error."""
        error = VideoNotOnPageError(PAGE, "x")
        assert handle_youtube_error(error, PAGE) is error


# --------------------------------------------------------------------------- resolver


def _resolve(info, url=PAGE):
    from services.url_resolver_service import resolve_video_url

    with patch("yt_dlp.YoutubeDL") as cls:
        ydl = MagicMock()
        cls.return_value.__enter__.return_value = ydl
        ydl.extract_info.return_value = info
        return resolve_video_url(url)


class TestResolverMarksAmbiguousEntries:
    def test_entries_sharing_the_page_url_carry_their_ids(self):
        page = _cnn_like_page()
        result = _resolve(page)
        assert result["type"] == "multiple"
        assert {v["url"] for v in result["videos"]} == {PAGE}
        assert [v["item_id"] for v in result["videos"]] == [
            e["id"] for e in page["entries"]
        ]

    def test_entries_with_their_own_urls_carry_no_id(self):
        info = {
            "extractor": "generic",
            "entries": [
                {"id": "a", "title": "A", "duration": 30, "url": "https://site/a"},
                {"id": "b", "title": "B", "duration": 40, "url": "https://site/b"},
            ],
        }
        result = _resolve(info, "https://site/article")
        assert all("item_id" not in v for v in result["videos"])

    def test_a_single_video_on_the_page_url_carries_its_id(self):
        page = _cnn_like_page(1)
        result = _resolve(page)
        assert result["type"] == "single"
        assert result["video"]["item_id"] == page["entries"][0]["id"]


# --------------------------------------------------------------------------- download


class PageYDL:
    """A yt-dlp stand-in for a page: honours match_filter, records the download."""

    page = None
    downloaded = None

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        assert download is False, "the service must resolve before downloading"
        keep = self.opts.get("match_filter")
        page = dict(PageYDL.page)
        page["entries"] = [e for e in page["entries"] if not keep or keep(e) is None]
        return page

    def process_ie_result(self, info, download=True):
        PageYDL.downloaded.append(info["id"])
        path = os.path.join(self.opts["outtmpl"].rsplit("/", 1)[0], "clip.mp4")
        with open(path, "wb") as handle:
            handle.write(b"video")
        return {**info, "requested_downloads": [{"filepath": path}]}

    def prepare_filename(self, info):
        return info["requested_downloads"][0]["filepath"]


@pytest.fixture
def page_service(monkeypatch, tmp_path):
    from config import get_config
    from services import youtube_service

    PageYDL.page = _cnn_like_page(10)
    PageYDL.downloaded = []
    work, final = tmp_path / "work", tmp_path / "final"
    work.mkdir()
    final.mkdir()

    real = get_config()
    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", PageYDL)
    monkeypatch.setattr(youtube_service, "DOWNLOADS_FOLDER", str(final))
    monkeypatch.setattr(
        youtube_service,
        "config",
        MagicMock(
            USE_FAKE_YTDLP=False,
            DOWNLOADS_FOLDER=str(final),
            FAST_WORK_DIR=str(work),
            YTDLP_OPTIMIZED_FORMAT=real.YTDLP_OPTIMIZED_FORMAT,
            YTDLP_MERGE_OUTPUT_FORMAT=real.YTDLP_MERGE_OUTPUT_FORMAT,
            YTDLP_EXTRACTOR_ARGS=real.YTDLP_EXTRACTOR_ARGS,
            DEBUG=False,
        ),
    )
    return youtube_service


class TestDownloadTakesOneVideo:
    def test_the_chosen_video_and_only_it_is_downloaded(self, page_service):
        chosen = PageYDL.page["entries"][7]
        path, metadata = page_service.download_youtube_video_with_progress(
            PAGE, item_id=chosen["id"]
        )
        assert PageYDL.downloaded == [chosen["id"]]
        assert metadata["title"] == "Clip 7"
        assert os.path.exists(path)

    def test_a_page_without_a_choice_downloads_nothing(self, page_service):
        with pytest.raises(PageHasMultipleVideosError):
            page_service.download_youtube_video_with_progress(PAGE)
        assert PageYDL.downloaded == []

    def test_the_pipeline_download_takes_the_chosen_video_too(self, page_service):
        chosen = PageYDL.page["entries"][2]
        _path, metadata = page_service.download_youtube_video(
            PAGE, item_id=chosen["id"]
        )
        assert PageYDL.downloaded == [chosen["id"]]
        assert metadata["title"] == "Clip 2"

    def test_the_pipeline_download_refuses_a_page_without_a_choice(self, page_service):
        with pytest.raises(PageHasMultipleVideosError):
            page_service.download_youtube_video(PAGE)
        assert PageYDL.downloaded == []


# --------------------------------------------------------------------------- metadata


class TestMetadataDescribesTheChosenVideo:
    def _service(self):
        from services.metadata_service import VideoMetadataService

        service = VideoMetadataService()
        service.cache_ttl = 3600
        return service

    def test_two_videos_from_one_page_are_cached_apart(self):
        service = self._service()
        page = _cnn_like_page()
        with patch("yt_dlp.YoutubeDL") as cls:

            def build(opts):
                ydl = MagicMock()
                keep = opts.get("match_filter")
                ydl.extract_info.return_value = {
                    **page,
                    "entries": [e for e in page["entries"] if keep(e) is None],
                }
                cls.return_value.__enter__.return_value = ydl
                return cls.return_value

            cls.side_effect = build
            first, _ = service.extract_metadata(PAGE, page["entries"][0]["id"])
            second, _ = service.extract_metadata(PAGE, page["entries"][2]["id"])

        assert (first.title, second.title) == ("Clip 0", "Clip 2")

    def test_a_page_without_a_choice_keeps_its_precise_code(self):
        from services.metadata_service import MetadataExtractionError

        with patch("yt_dlp.YoutubeDL") as cls:
            ydl = MagicMock()
            cls.return_value.__enter__.return_value = ydl
            ydl.extract_info.return_value = _cnn_like_page()
            with pytest.raises(MetadataExtractionError) as caught:
                self._service().extract_metadata(PAGE)

        assert caught.value.error_code == "PAGE_HAS_MULTIPLE_VIDEOS"
        assert caught.value.recoverable is False
