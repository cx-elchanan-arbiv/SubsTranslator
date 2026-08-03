"""
Unit tests for the subtitle-quality flag plumbing.

Two things are covered, and both are things that break silently rather than loudly:

1. **Request parsing** — the four toggles arrive as multipart form STRINGS from ``/upload``
   and as real JSON booleans from ``/youtube``. A bare truthiness check would read the
   string ``"false"`` as ON. These tests assert the exact kwargs handed to
   ``apply_async``, i.e. what the Celery task will actually receive.
2. **Cue normalisation** — the three producers in the pipeline use three different shapes
   (legacy ``translated_text``, ``words_to_cues`` with no translation, ``translate_cues``
   with ``translated``). ``normalize_cues`` is what lets any stage consume any
   predecessor's output, which is what makes all eight flag combinations work.

No network, no Celery broker, no ffmpeg: ``apply_async`` is mocked and its kwargs captured.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Must be set BEFORE `app` is imported (see the flask_client fixture). `app` starts the
# token cleanup scheduler, which skips its non-daemon `threading.Timer` only when TESTING
# is exactly "true" — with any other value the timer keeps the interpreter alive and the
# whole pytest process hangs at shutdown. tests/unit/test_api_contracts.py does the same.
os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")


def _find_backend_dir():
    """
    Locate the backend package root.

    Searched rather than computed from a fixed relative path because docker-compose mounts
    ``./tests`` over ``/app/tests``: inside the container this file runs as
    ``/app/tests/unit/...`` with ``/app`` as the backend root.
    """
    for seed in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = seed
        while True:
            if os.path.isfile(os.path.join(path, "services", "subtitle_pipeline.py")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise RuntimeError("could not locate the backend directory containing services/")


backend_dir = _find_backend_dir()
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.subtitle_pipeline import (  # noqa: E402
    FLAG_DEFAULTS,
    STYLE_CLEAN,
    STYLE_FAITHFUL,
    any_enabled,
    flags_summary,
    normalize_cues,
    parse_bool,
    parse_subtitle_flags,
    parse_translation_style,
    resolve_flags,
)

ALL_FLAG_KEYS = {"spotting_v2", "translation_v2", "translation_style", "render_v2"}


# =====================================================================================
# Value parsing
# =====================================================================================
@pytest.mark.unit
class TestParseBool:
    @pytest.mark.parametrize(
        "value", [True, "true", "True", "TRUE", "1", 1, "yes", "on", "y", "t"]
    )
    def test_truthy_forms(self, value):
        assert parse_bool(value) is True

    @pytest.mark.parametrize(
        "value", [False, "false", "False", "0", 0, "no", "off", "", "none", "null"]
    )
    def test_falsy_forms(self, value):
        assert parse_bool(value) is False

    def test_the_string_false_is_not_truthy(self):
        """The bug this function exists to prevent: `if request.form.get(...)`."""
        assert bool("false") is True  # the trap
        assert parse_bool("false") is False  # the fix

    def test_missing_value_uses_the_default(self):
        assert parse_bool(None) is False
        assert parse_bool(None, default=True) is True

    def test_garbage_falls_back_instead_of_raising(self):
        assert parse_bool("maybe") is False
        assert parse_bool(["true"]) is False


@pytest.mark.unit
class TestParseTranslationStyle:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("clean", STYLE_CLEAN),
            ("CLEAN", STYLE_CLEAN),
            (" faithful ", STYLE_FAITHFUL),
            ("faithful", STYLE_FAITHFUL),
        ],
    )
    def test_valid_styles(self, value, expected):
        assert parse_translation_style(value) == expected

    @pytest.mark.parametrize("value", [None, "", "verbatim", "literal", 7])
    def test_invalid_styles_fall_back_to_clean(self, value):
        assert parse_translation_style(value) == STYLE_CLEAN


@pytest.mark.unit
class TestParseSubtitleFlags:
    def test_empty_request_is_all_off(self):
        assert parse_subtitle_flags({}) == FLAG_DEFAULTS
        assert parse_subtitle_flags(None) == FLAG_DEFAULTS

    def test_form_strings(self):
        flags = parse_subtitle_flags(
            {
                "spotting_v2": "true",
                "translation_v2": "false",
                "translation_style": "faithful",
                "render_v2": "true",
            }
        )
        assert flags == {
            "spotting_v2": True,
            "translation_v2": False,
            "translation_style": "faithful",
            "render_v2": True,
        }

    def test_json_booleans(self):
        flags = parse_subtitle_flags(
            {"spotting_v2": False, "translation_v2": True, "render_v2": False}
        )
        assert flags == {
            "spotting_v2": False,
            "translation_v2": True,
            "translation_style": "clean",
            "render_v2": False,
        }

    def test_always_returns_all_four_keys_with_correct_types(self):
        flags = parse_subtitle_flags({"spotting_v2": "1"})
        assert set(flags) == ALL_FLAG_KEYS
        for name in ("spotting_v2", "translation_v2", "render_v2"):
            assert isinstance(flags[name], bool)
        assert isinstance(flags["translation_style"], str)

    def test_flags_are_independent(self):
        """Turning one on must not turn any other on."""
        for name in ("spotting_v2", "translation_v2", "render_v2"):
            flags = parse_subtitle_flags({name: "true"})
            assert flags[name] is True
            others = [
                k for k in ("spotting_v2", "translation_v2", "render_v2") if k != name
            ]
            assert all(flags[other] is False for other in others)


@pytest.mark.unit
class TestResolveFlagsAndSummary:
    def test_no_arguments_is_all_off(self):
        assert resolve_flags() == FLAG_DEFAULTS

    def test_reparses_values_that_travelled_through_celery(self):
        assert resolve_flags(
            spotting_v2="true",
            translation_v2=0,
            translation_style="FAITHFUL",
            render_v2=True,
        ) == {
            "spotting_v2": True,
            "translation_v2": False,
            "translation_style": "faithful",
            "render_v2": True,
        }

    def test_any_enabled_ignores_style(self):
        assert any_enabled(FLAG_DEFAULTS) is False
        assert any_enabled(resolve_flags(translation_style="faithful")) is False
        assert any_enabled(resolve_flags(render_v2=True)) is True

    def test_summary_names_every_flag_on_one_line(self):
        summary = flags_summary(
            resolve_flags(spotting_v2=True, translation_style="faithful")
        )
        assert "\n" not in summary
        assert summary == (
            "spotting_v2=True translation_v2=False "
            "translation_style=faithful render_v2=False"
        )


# =====================================================================================
# Cue normalisation — the contract that makes the stages composable
# =====================================================================================
@pytest.mark.unit
class TestNormalizeCues:
    def test_legacy_segments_pass_through(self):
        cues = normalize_cues(
            [{"start": 0.0, "end": 2.0, "text": "Hello", "translated_text": "שלום"}]
        )
        assert cues == [
            {"start": 0.0, "end": 2.0, "text": "Hello", "translated_text": "שלום"}
        ]

    def test_words_to_cues_output_gets_an_empty_translation(self):
        """subtitle_engine emits no translation key at all; the shape must still be complete."""
        cues = normalize_cues(
            [{"start": 1.0, "end": 3.5, "text": "It could complicate things."}]
        )
        assert cues == [
            {
                "start": 1.0,
                "end": 3.5,
                "text": "It could complicate things.",
                "translated_text": "",
            }
        ]

    def test_translation_v2_translated_key_is_mapped(self):
        """translate_cues() writes "translated"; the app speaks "translated_text"."""
        cues = normalize_cues(
            [{"start": 0, "end": 1, "text": "Yeah.", "translated": "כן."}]
        )
        assert cues[0]["translated_text"] == "כן."
        assert "translated" not in cues[0]

    def test_existing_translated_text_wins_over_translated(self):
        cues = normalize_cues(
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "a",
                    "translated_text": "first",
                    "translated": "second",
                }
            ]
        )
        assert cues[0]["translated_text"] == "first"

    def test_blank_translated_text_does_not_beat_a_real_translation(self):
        """
        Regression: normalising twice used to silently discard every v2 translation.

        Pass 1 (after transcription) sets ``translated_text: ""``. ``translate_cues`` then
        copies that dict and adds its result under ``translated``. If pass 2 preferred the
        merely-present empty ``translated_text``, the job would burn in the SOURCE text and
        still report success — the worst possible failure mode for a quality flag.
        """
        after_translation = {
            "start": 0.0,
            "end": 2.0,
            "text": "It could complicate things.",
            "translated_text": "",  # left by the previous normalize_cues pass
            "translated": "זה עלול לסבך את העניינים.",  # written by translate_cues
        }
        cues = normalize_cues([after_translation])
        assert cues[0]["translated_text"] == "זה עלול לסבך את העניינים."

    def test_whitespace_only_translated_text_does_not_beat_a_real_translation(self):
        cues = normalize_cues(
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "a",
                    "translated_text": "   ",
                    "translated": "כן.",
                }
            ]
        )
        assert cues[0]["translated_text"] == "כן."

    def test_output_is_a_plain_list_of_plain_dicts(self):
        """translate_cues returns a TranslationResult subclass; downstream code copies dicts."""
        from services.translation_v2 import TranslationResult

        result = TranslationResult(
            [{"start": 0, "end": 1, "text": "a", "translated": "b"}]
        )
        cues = normalize_cues(result)
        assert type(cues) is list
        assert all(type(cue) is dict for cue in cues)

    def test_times_are_coerced_to_float(self):
        cues = normalize_cues([{"start": "1.5", "end": 3, "text": "x"}])
        assert cues[0]["start"] == 1.5
        assert cues[0]["end"] == 3.0
        assert isinstance(cues[0]["start"], float)
        assert isinstance(cues[0]["end"], float)

    def test_unparsable_times_become_zero_rather_than_raising(self):
        cues = normalize_cues([{"start": None, "end": "later", "text": "x"}])
        assert cues[0]["start"] == 0.0
        assert cues[0]["end"] == 0.0

    def test_whitespace_is_stripped(self):
        cues = normalize_cues(
            [{"start": 0, "end": 1, "text": "  padded  ", "translated": " כן "}]
        )
        assert cues[0]["text"] == "padded"
        assert cues[0]["translated_text"] == "כן"

    def test_fully_blank_cues_are_dropped(self):
        cues = normalize_cues(
            [
                {"start": 0, "end": 1, "text": "   "},
                {"start": 1, "end": 2, "text": "", "translated": ""},
                {"start": 2, "end": 3, "text": "keep me"},
            ]
        )
        assert [cue["text"] for cue in cues] == ["keep me"]

    def test_translation_only_cue_is_kept(self):
        cues = normalize_cues(
            [{"start": 0, "end": 1, "text": "", "translated": "שלום"}]
        )
        assert len(cues) == 1
        assert cues[0]["translated_text"] == "שלום"

    def test_non_dict_entries_are_skipped(self):
        cues = normalize_cues(
            [None, "nonsense", 42, {"start": 0, "end": 1, "text": "ok"}]
        )
        assert [cue["text"] for cue in cues] == ["ok"]

    def test_empty_and_none_input(self):
        assert normalize_cues([]) == []
        assert normalize_cues(None) == []

    def test_input_order_is_preserved(self):
        cues = normalize_cues(
            [
                {"start": 5, "end": 6, "text": "second"},
                {"start": 0, "end": 1, "text": "first"},
            ]
        )
        assert [cue["text"] for cue in cues] == ["second", "first"]

    def test_input_is_not_mutated(self):
        source = [{"start": 0, "end": 1, "text": "a", "translated": "b"}]
        normalize_cues(source)
        assert source == [{"start": 0, "end": 1, "text": "a", "translated": "b"}]

    # -- unknown keys survive the normaliser ---------------------------------------
    def test_unknown_keys_are_carried_through(self):
        """Normalising is about agreeing on four keys, not about being an allow-list.

        This function sits between every producer and every consumer in the pipeline. If
        it drops what it does not recognise, then a stage that starts emitting, say, a
        speaker label finds it silently deleted by the plumbing rather than by anyone's
        decision — and the loss is invisible, because the four keys it does know about
        are all still there.
        """
        cues = normalize_cues(
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "hello",
                    "speaker": "HOST",
                    "confidence": 0.93,
                    "words": [{"w": "hello", "s": 0.0, "e": 1.0}],
                }
            ]
        )
        assert cues[0]["speaker"] == "HOST"
        assert cues[0]["confidence"] == 0.93
        assert cues[0]["words"] == [{"w": "hello", "s": 0.0, "e": 1.0}]

    def test_the_four_known_keys_are_still_normalised_around_them(self):
        """Passing extras through must not stop the normalisation itself happening."""
        cues = normalize_cues(
            [
                {
                    "start": "2.5",
                    "end": 4,
                    "text": " x ",
                    "translated": " כן ",
                    "extra": 1,
                }
            ]
        )
        assert cues[0] == {
            "start": 2.5,
            "end": 4.0,
            "text": "x",
            "translated_text": "כן",
            "extra": 1,
        }

    def test_translated_is_consumed_and_not_left_behind(self):
        """``translated`` is the one key deliberately dropped.

        It is folded into ``translated_text`` so no downstream code has to know which of
        the two spellings it is looking at; leaving both would reintroduce exactly that
        ambiguity.
        """
        cues = normalize_cues([{"start": 0, "end": 1, "text": "a", "translated": "ב"}])
        assert cues[0]["translated_text"] == "ב"
        assert "translated" not in cues[0]

    def test_an_unknown_key_never_overwrites_a_normalised_one(self):
        """A stage emitting a stray `text` variant must not win over the real field."""
        cues = normalize_cues(
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "real",
                    "translated_text": "תרגום",
                    "note": "x",
                }
            ]
        )
        assert cues[0]["text"] == "real"
        assert cues[0]["translated_text"] == "תרגום"
        assert cues[0]["note"] == "x"

    def test_extras_survive_a_reflow_pass(self):
        """End to end with the pass that runs straight after it in the v2 branch."""
        from services.subtitle_engine import reflow_dangling_connectors

        cues = normalize_cues(
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "a",
                    "translated": "המטוס נחת ו",
                    "speaker": "A",
                },
                {
                    "start": 1.2,
                    "end": 2,
                    "text": "b",
                    "translated": "הנוסעים ירדו",
                    "speaker": "B",
                },
            ]
        )
        out = reflow_dangling_connectors(cues)
        assert out[1]["translated_text"] == "והנוסעים ירדו"
        assert [cue["speaker"] for cue in out] == ["A", "B"]

    def test_round_trips_a_real_words_to_cues_result(self):
        """End-to-end shape check against the actual engine, not a hand-written stub."""
        from services.subtitle_engine import words_to_cues

        cues = words_to_cues(
            [
                {"s": 0.0, "e": 0.4, "w": "It"},
                {"s": 0.4, "e": 0.8, "w": "could"},
                {"s": 0.8, "e": 1.4, "w": "complicate"},
                {"s": 1.4, "e": 1.8, "w": "things."},
            ]
        )
        normalized = normalize_cues(cues)
        assert normalized
        for cue in normalized:
            # The four normalised keys, plus whatever the producer added. `word_stats`
            # is `words_to_cues`'s degeneracy read-out and the hallucination gate reads
            # it downstream of this function — a normaliser that dropped it would be an
            # allow-list, which is exactly what `normalize_cues` documents it is not.
            assert {"start", "end", "text", "translated_text"} <= set(cue)
            assert set(cue) - {"start", "end", "text", "translated_text"} == {
                "word_stats"
            }


# =====================================================================================
# Route parsing -> Celery kwargs
# =====================================================================================
@pytest.fixture
def flask_client():
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def upload_route(tmp_path):
    """Patch out everything /upload does except parsing: no disk probe, no Celery."""
    from api import video_routes

    task = MagicMock()
    task.apply_async.return_value = MagicMock(id="task-123")
    original_upload_folder = video_routes.config.UPLOAD_FOLDER
    video_routes.config.UPLOAD_FOLDER = str(tmp_path)
    try:
        with (
            patch.object(video_routes, "process_video_task", task),
            patch.object(
                video_routes, "probe_file_safe", return_value=({"duration": 1.0}, None)
            ),
        ):
            yield task
    finally:
        video_routes.config.UPLOAD_FOLDER = original_upload_folder


@pytest.fixture
def youtube_route():
    """Patch out everything /youtube does except parsing."""
    from api import video_routes

    task = MagicMock()
    task.apply_async.return_value = MagicMock(id="task-456")
    with (
        patch.object(video_routes, "download_and_process_youtube_task", task),
        patch.object(video_routes.config, "is_youtube_restricted", return_value=False),
    ):
        yield task


def _captured_kwargs(task):
    assert task.apply_async.called, "the route did not enqueue a task"
    return task.apply_async.call_args.kwargs["kwargs"]


def _video_file():
    import io

    return (io.BytesIO(b"\x00" * 2048), "clip.mp4")


@pytest.mark.unit
class TestUploadRouteFlagParsing:
    def test_defaults_when_the_client_sends_nothing(self, flask_client, upload_route):
        response = flask_client.post(
            "/upload",
            data={"file": _video_file(), "target_lang": "he", "whisper_model": "base"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 202
        assert _captured_kwargs(upload_route) == FLAG_DEFAULTS

    def test_all_flags_on(self, flask_client, upload_route):
        response = flask_client.post(
            "/upload",
            data={
                "file": _video_file(),
                "target_lang": "he",
                "whisper_model": "base",
                "spotting_v2": "true",
                "translation_v2": "true",
                "translation_style": "faithful",
                "render_v2": "true",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 202
        assert _captured_kwargs(upload_route) == {
            "spotting_v2": True,
            "translation_v2": True,
            "translation_style": "faithful",
            "render_v2": True,
        }

    def test_the_string_false_stays_off(self, flask_client, upload_route):
        """Regression guard: form booleans are strings, and "false" is truthy in Python."""
        response = flask_client.post(
            "/upload",
            data={
                "file": _video_file(),
                "target_lang": "he",
                "whisper_model": "base",
                "spotting_v2": "false",
                "translation_v2": "false",
                "render_v2": "false",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 202
        kwargs = _captured_kwargs(upload_route)
        assert kwargs["spotting_v2"] is False
        assert kwargs["translation_v2"] is False
        assert kwargs["render_v2"] is False

    def test_one_flag_at_a_time(self, flask_client, upload_route):
        for name in ("spotting_v2", "translation_v2", "render_v2"):
            upload_route.apply_async.reset_mock()
            response = flask_client.post(
                "/upload",
                data={
                    "file": _video_file(),
                    "target_lang": "he",
                    "whisper_model": "base",
                    name: "true",
                },
                content_type="multipart/form-data",
            )
            assert response.status_code == 202
            kwargs = _captured_kwargs(upload_route)
            assert kwargs[name] is True, name
            assert (
                sum(
                    1
                    for k in ("spotting_v2", "translation_v2", "render_v2")
                    if kwargs[k]
                )
                == 1
            )


@pytest.mark.unit
class TestYoutubeRouteFlagParsing:
    URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_defaults_when_the_client_sends_nothing(self, flask_client, youtube_route):
        response = flask_client.post(
            "/youtube",
            json={"url": self.URL, "target_lang": "he", "whisper_model": "base"},
        )
        assert response.status_code == 202
        assert _captured_kwargs(youtube_route) == FLAG_DEFAULTS

    def test_json_booleans(self, flask_client, youtube_route):
        response = flask_client.post(
            "/youtube",
            json={
                "url": self.URL,
                "target_lang": "he",
                "whisper_model": "base",
                "spotting_v2": True,
                "translation_v2": True,
                "translation_style": "clean",
                "render_v2": True,
            },
        )
        assert response.status_code == 202
        assert _captured_kwargs(youtube_route) == {
            "spotting_v2": True,
            "translation_v2": True,
            "translation_style": "clean",
            "render_v2": True,
        }

    def test_json_false_stays_off(self, flask_client, youtube_route):
        response = flask_client.post(
            "/youtube",
            json={
                "url": self.URL,
                "target_lang": "he",
                "whisper_model": "base",
                "spotting_v2": False,
                "render_v2": False,
            },
        )
        assert response.status_code == 202
        kwargs = _captured_kwargs(youtube_route)
        assert kwargs["spotting_v2"] is False
        assert kwargs["render_v2"] is False

    def test_formdata_strings(self, flask_client, youtube_route):
        """The custom-logo path sends /youtube as multipart, so strings must work too."""
        response = flask_client.post(
            "/youtube",
            data={
                "url": self.URL,
                "target_lang": "he",
                "whisper_model": "base",
                "spotting_v2": "true",
                "translation_v2": "false",
                "translation_style": "faithful",
                "render_v2": "true",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 202
        assert _captured_kwargs(youtube_route) == {
            "spotting_v2": True,
            "translation_v2": False,
            "translation_style": "faithful",
            "render_v2": True,
        }

    def test_positional_args_are_unchanged(self, flask_client, youtube_route):
        """The flags ride in kwargs; the legacy positional contract must not shift."""
        flask_client.post(
            "/youtube",
            json={"url": self.URL, "target_lang": "he", "whisper_model": "base"},
        )
        args = youtube_route.apply_async.call_args.kwargs["args"]
        assert args[0] == self.URL
        assert (
            len(args) == 7
        )  # url, source, target, auto_create, model, service, watermark


@pytest.mark.unit
class TestTaskSignaturesAcceptTheFlags:
    """The chain is only as good as its narrowest link: every hop must take all four."""

    @pytest.mark.parametrize(
        "module,name",
        [
            ("tasks.processing_tasks", "process_video_task"),
            ("tasks.download_tasks", "download_and_process_youtube_task"),
        ],
    )
    def test_task_accepts_every_flag_as_a_keyword(self, module, name):
        import importlib
        import inspect

        task = getattr(importlib.import_module(module), name)
        params = inspect.signature(task.run).parameters
        for flag in ALL_FLAG_KEYS:
            assert flag in params, f"{name} does not accept {flag}"
            assert (
                params[flag].default == FLAG_DEFAULTS[flag]
            ), f"{name}.{flag} must default to the legacy behaviour"

    @staticmethod
    def _run_youtube_task(**task_kwargs):
        """Drive ``download_and_process_youtube_task`` with the download stubbed out.

        Returns ``(chained_kwargs, result)`` — the kwargs handed to
        ``process_video_task.apply_async``, and the task's own return value.
        """
        from tasks import download_tasks

        chained = {}

        def fake_apply_async(args=None, kwargs=None, queue=None):
            chained["args"] = args
            chained["kwargs"] = kwargs
            return MagicMock(id="chained-task-1")

        with (
            patch.object(download_tasks, "yt_dlp") as fake_ytdlp,
            patch.object(download_tasks, "download_youtube_video") as fake_download,
            patch.object(
                download_tasks.process_video_task, "apply_async", fake_apply_async
            ),
            patch.object(download_tasks.time, "sleep", lambda *_a, **_k: None),
            patch.object(
                download_tasks.download_and_process_youtube_task,
                "update_state",
                MagicMock(),
            ),
        ):
            fake_ytdlp.YoutubeDL.return_value.__enter__.return_value.extract_info.return_value = {
                "title": "clip",
                "duration": 30,
            }
            fake_download.return_value = ("/tmp/clip.mp4", {"title": "clip"})

            task = download_tasks.download_and_process_youtube_task
            task.push_request(id="yt-task-1")
            try:
                result = task.run(
                    "https://youtu.be/abc",
                    "en",
                    "he",
                    True,
                    "tiny",
                    **task_kwargs,
                )
            finally:
                task.pop_request()
        return chained, result

    def test_youtube_task_forwards_every_flag_to_the_processing_task(self):
        """A flag parsed by the route but dropped by the chain is invisible in the UI.

        Asserted on the kwargs ``apply_async`` actually receives rather than on the shape
        of the source text: the previous version of this test matched the literal
        ``"flag": flag`` and went red the moment the same four values were forwarded as a
        dict, which is a refactor, not a regression. What matters is what the downstream
        task is handed.
        """
        chained, _result = self._run_youtube_task(
            spotting_v2=True,
            translation_v2=True,
            translation_style=STYLE_FAITHFUL,
            render_v2=True,
        )

        assert chained.get("kwargs"), "the processing task was never enqueued"
        assert (
            set(chained["kwargs"]) == ALL_FLAG_KEYS
        ), f"chained kwargs are {sorted(chained['kwargs'])}, expected exactly the flags"
        assert chained["kwargs"] == {
            "spotting_v2": True,
            "translation_v2": True,
            "translation_style": STYLE_FAITHFUL,
            "render_v2": True,
        }

    def test_youtube_task_forwards_the_defaults_when_nothing_is_set(self):
        """All four off is the legacy pipeline, and must survive the hop unchanged."""
        chained, _result = self._run_youtube_task()
        assert chained["kwargs"] == FLAG_DEFAULTS

    def test_youtube_task_normalises_flags_before_forwarding(self):
        """A flag that arrives as a string must not reach the task as a string.

        Celery serialises through JSON, and a route or a retry can hand this task the
        string ``"true"``. Forwarding it unresolved would make ``if spotting_v2`` true for
        ``"false"`` as well.
        """
        chained, _result = self._run_youtube_task(
            spotting_v2="true",
            translation_v2="false",
            translation_style="nonsense",
            render_v2=1,
        )
        assert chained["kwargs"]["spotting_v2"] is True
        assert chained["kwargs"]["translation_v2"] is False
        assert chained["kwargs"]["render_v2"] is True
        assert (
            chained["kwargs"]["translation_style"] == STYLE_CLEAN
        ), "an unrecognised style must fall back to the default, not be passed through"

    def test_youtube_task_reports_the_flags_as_user_choices(self):
        """The YouTube path must be attributable to its settings exactly like an upload."""
        chained, result = self._run_youtube_task(
            spotting_v2=True, translation_style=STYLE_FAITHFUL
        )

        # ...in the payload handed to the processing task,
        processing_info = chained["args"][8]
        choices = processing_info["user_choices"]
        assert choices["spotting_v2"] is True
        assert choices["translation_style"] == STYLE_FAITHFUL
        assert set(ALL_FLAG_KEYS).issubset(choices)

        # ...and in what this task returns to the API.
        assert result["user_choices"]["spotting_v2"] is True
        assert set(ALL_FLAG_KEYS).issubset(result["user_choices"])


# ======================================================================================
# P2 — glossary plumbing
# ======================================================================================
@pytest.mark.unit
class TestParseGlossary:
    """The glossary reaches the pipeline through an HTML form, a JSON body or a Celery
    kwarg, so all three shapes have to arrive as the same dict — and a malformed one
    may never fail a job."""

    def test_none_and_blank(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary(None) == {}
        assert parse_glossary("") == {}
        assert parse_glossary("   ") == {}

    def test_real_mapping(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary({"Ivrit": "עִברית"}) == {"Ivrit": "עִברית"}

    def test_json_string_from_a_form_field(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary('{"Ivrit": "עִברית"}') == {"Ivrit": "עִברית"}

    def test_invalid_json_is_ignored_not_raised(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary("{not json") == {}

    def test_json_of_the_wrong_type_is_ignored(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary("[1, 2, 3]") == {}
        assert parse_glossary("42") == {}

    def test_values_are_stringified_and_stripped(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary({" Ivrit ": " עִברית "}) == {"Ivrit": "עִברית"}

    def test_blank_and_null_entries_are_dropped(self):
        from services.subtitle_pipeline import parse_glossary

        assert parse_glossary({"Ivrit": "עִברית", "": "x", "y": "", "z": None}) == {
            "Ivrit": "עִברית"
        }

    def test_never_raises_on_anything(self):
        from services.subtitle_pipeline import parse_glossary

        for value in (object(), 3.14, True, [], set()):
            assert isinstance(parse_glossary(value), dict)
