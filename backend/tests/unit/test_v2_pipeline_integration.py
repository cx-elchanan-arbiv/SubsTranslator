"""
Integration tests for ``process_video_task`` across every subtitle-quality flag combination.

Why this file exists
--------------------
The four toggles in :mod:`services.subtitle_pipeline` produce eight pipeline shapes, each
of which can additionally run with or without a watermark. The stages are written to be
independent, but "independent" is a claim about the wiring, and the wiring is exactly what
nothing else tests: the unit tests prove each stage is correct in isolation, and the live
E2E run exercises one combination at a time at a cost of minutes. A cue-shape mismatch
between two stages — the sort of defect where ``normalize_cues`` hands on a key the next
stage does not read — passes every unit test and produces a video full of blank subtitles.

That is not hypothetical. It is the bug this file was written around: ``create_srt_file``
tested its text with ``if text is None``, while ``normalize_cues`` emits
``translated_text=""`` rather than omitting the key, so the ``spotting_v2``-on /
``translation_v2``-off combination wrote a subtitle file of empty cues *and reported
success*. See :class:`TestTheContentAssertionsActuallyCatchTheRegression`, which reverts
the fix in-place and proves these tests go red.

Method
------
Stage boundaries are mocked — transcription, translation and FFmpeg never run — but
everything *between* them is the real code: ``words_to_cues``, ``normalize_cues``,
``reflow_dangling_connectors``, ``cps_report`` and, above all, ``create_srt_file``, whose
output is read back off disk and asserted on. Assertions are therefore in two layers:

1. **Which stages ran** — the right transcriber, the right translator, the right renderer
   for each combination, and no trace of the other branch.
2. **What came out** — the actual bytes of both SRT files.

No network, no Celery broker, no ffmpeg.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")


def _find_backend_dir():
    """docker-compose mounts ./tests over /app/tests, so a fixed relative path is wrong."""
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

from services.subtitle_pipeline import STYLE_CLEAN, STYLE_FAITHFUL  # noqa: E402
from services.translation_v2 import TokenUsage, TranslationResult  # noqa: E402

TASK_MODULE = "tasks.processing_tasks"


# ----------------------------------------------------------------------------------
# fake stage outputs
# ----------------------------------------------------------------------------------
#: Whisper word timestamps. Punctuated, so `words_to_cues` takes its normal path, and
#: long enough that it produces more than one cue.
WORDS = [
    {"s": 0.0, "e": 0.35, "w": "It"},
    {"s": 0.35, "e": 0.70, "w": "could"},
    {"s": 0.70, "e": 1.30, "w": "complicate"},
    {"s": 1.30, "e": 1.90, "w": "things."},
    {"s": 2.10, "e": 2.40, "w": "Do"},
    {"s": 2.40, "e": 2.65, "w": "you"},
    {"s": 2.65, "e": 3.10, "w": "worry"},
    {"s": 3.10, "e": 3.40, "w": "about"},
    {"s": 3.40, "e": 3.80, "w": "that?"},
    {"s": 4.20, "e": 4.60, "w": "Yeah,"},
    {"s": 4.60, "e": 4.90, "w": "I"},
    {"s": 4.90, "e": 5.30, "w": "think"},
    {"s": 5.30, "e": 5.60, "w": "about"},
    {"s": 5.60, "e": 6.00, "w": "it."},
]

#: Whisper's own speech segments — what the pipeline uses when spotting_v2 is OFF.
#: Deliberately cut the way Whisper really cuts: mid-clause, with a sentence ending in
#: the middle of a segment and a question glued to its answer. These are the defects
#: ``words_to_cues`` exists to repair, so re-spotting the same words must produce
#: visibly different cue boundaries.
WHISPER_SEGMENTS = [
    {"start": 0.0, "end": 1.3, "text": "It could complicate"},
    {"start": 1.3, "end": 3.8, "text": "things. Do you worry about that?"},
    {"start": 4.2, "end": 6.0, "text": "Yeah, I think about it."},
]

#: What the legacy streaming transcriber returns: segments already carrying a translation.
STREAMED_SEGMENTS = [
    {**segment, "translated_text": f"[legacy-he] {segment['text']}"}
    for segment in WHISPER_SEGMENTS
]

HEBREW = {
    "It could complicate things.": "זה עלול לסבך את העניינים.",
    "Do you worry about that?": "אתה דואג מזה?",
    "Yeah, I think about it.": "כן, אני חושב על זה.",
}


def _translate(text):
    """Deterministic stand-in for a real translation, marked so it is unmistakable."""
    return HEBREW.get(text.strip(), f"[he] {text.strip()}")


class StageSpy:
    """Records which pipeline stages ran, and with what."""

    def __init__(self):
        self.calls = {}

    def record(self, name, **details):
        self.calls.setdefault(name, []).append(details)

    def ran(self, name):
        return name in self.calls

    @property
    def names(self):
        return set(self.calls)


@pytest.fixture()
def pipeline(tmp_path):
    """Run ``process_video_task`` with every stage boundary mocked.

    Yields a ``run(**flags)`` callable returning ``(result, spy, downloads_dir)``.
    """
    # One output directory PER RUN. Several tests call run() twice and compare the two
    # results; a shared directory means the second run silently overwrites the first
    # one's SRT files and the comparison comes out equal no matter what the code does.
    run_counter = {"n": 0}
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 1024)
    logo_path = tmp_path / "logo.png"
    logo_path.write_bytes(b"\x89PNG" + b"\x00" * 64)

    spy = StageSpy()

    # --- transcription ------------------------------------------------------------
    def fake_transcribe_with_words(path, **kwargs):
        spy.record("transcribe_with_words", collect_words=kwargs.get("collect_words"))
        # Mirrors the real contract: words only when they were asked for.
        return {
            "language": "en",
            "segments": [dict(s) for s in WHISPER_SEGMENTS],
            "words": [dict(w) for w in WORDS] if kwargs.get("collect_words") else [],
        }

    def fake_transcribe_and_translate_streamed(path, **kwargs):
        spy.record("transcribe_and_translate_streamed")
        return {"language": "en", "segments": [dict(s) for s in STREAMED_SEGMENTS]}

    def fake_transcribe_video(path, **kwargs):
        spy.record("transcribe_video")
        return {"language": "en", "segments": [dict(s) for s in WHISPER_SEGMENTS]}

    # --- translation --------------------------------------------------------------
    def fake_translate_cues(cues, target_lang, style=None, progress_callback=None, **kw):
        spy.record(
            "translate_cues",
            target_lang=target_lang,
            style=style,
            cues=len(cues),
            has_progress_callback=progress_callback is not None,
        )
        if progress_callback:  # the v2 path must drive the UI during a long translation
            progress_callback(0, len(cues), "Translating cues 1-N")
            progress_callback(len(cues), len(cues), "Translated N cues")
        usage = TokenUsage()
        usage.add(1000, 500, "gpt-4o")
        return TranslationResult(
            [{**cue, "translated": _translate(cue["text"])} for cue in cues], usage
        )

    def fake_enforce_cps(cues, progress_callback=None, **kwargs):
        spy.record(
            "enforce_cps", has_progress_callback=progress_callback is not None
        )
        if progress_callback:
            progress_callback(0, len(cues), "Condensing over-long cues")
        usage = TokenUsage()
        usage.add(200, 100, "gpt-4o")
        return TranslationResult([dict(cue) for cue in cues], usage)

    def fake_translate_segments(segments, target_lang, **kwargs):
        spy.record("translate_segments", target_lang=target_lang)
        return [
            {**segment, "translated_text": f"[legacy-he] {segment['text']}"}
            for segment in segments
        ]

    # --- rendering ----------------------------------------------------------------
    def _touch(path):
        with open(path, "wb") as handle:
            handle.write(b"\x00" * 2048)

    def fake_create_video_with_ass(video, segments, output, **kwargs):
        spy.record(
            "create_video_with_ass",
            cues=len(segments),
            use_translation=kwargs.get("use_translation"),
            watermark_path=kwargs.get("watermark_path"),
        )
        _touch(output)
        return True

    def fake_create_video_with_subtitles(video, srt, output, language, **kwargs):
        spy.record("create_video_with_subtitles", srt=os.path.basename(srt))
        _touch(output)
        return True

    def fake_combined(video, srt, output, watermark, language, **kwargs):
        spy.record(
            "create_video_with_subtitles_and_watermark",
            srt=os.path.basename(srt),
            watermark=os.path.basename(watermark),
        )
        _touch(output)
        return True

    saved_stats = []

    def run(
        *,
        spotting_v2=False,
        translation_v2=False,
        render_v2=False,
        translation_style=STYLE_CLEAN,
        watermark=False,
        target_lang="he",
        auto_create_video=True,
        whisper_model="tiny",
        processing_info=None,
        translate_cues_impl=None,
    ):
        """``translate_cues_impl`` replaces the translation stage for this run.

        It has to be threaded through here rather than patched around the call: the
        ``patch.multiple`` below rebinds ``translate_cues`` itself, so an outer patch of
        the same name is silently overwritten.
        """
        from tasks import processing_tasks
        from services import subtitle_service as subtitle_service_module

        run_counter["n"] += 1
        downloads = tmp_path / f"downloads_{run_counter['n']}"
        downloads.mkdir()

        service = subtitle_service_module.subtitle_service
        watermark_config = (
            {"enabled": True, "custom_logo_path": str(logo_path), "position": "bottom-right"}
            if watermark
            else None
        )

        with patch.multiple(
            processing_tasks,
            DOWNLOADS_FOLDER=str(downloads),
            transcribe_with_words=fake_transcribe_with_words,
            transcribe_and_translate_streamed=fake_transcribe_and_translate_streamed,
            transcribe_video=fake_transcribe_video,
            translate_cues=translate_cues_impl or fake_translate_cues,
            enforce_cps=fake_enforce_cps,
            translate_segments=fake_translate_segments,
            save_video_stats=lambda payload: saved_stats.append(payload),
        ), patch.object(
            processing_tasks.time, "sleep", lambda *_a, **_k: None
        ), patch.object(
            service, "create_video_with_ass", fake_create_video_with_ass
        ), patch.object(
            service, "create_video_with_subtitles", fake_create_video_with_subtitles
        ), patch.object(
            service, "create_video_with_subtitles_and_watermark", fake_combined
        ), patch.object(
            processing_tasks.process_video_task, "update_state", MagicMock()
        ):
            # A real request id: the task slices it for a log line, and None would make
            # the stats block die in a way that masks genuine failures.
            processing_tasks.process_video_task.push_request(id="test-task-000000")
            try:
                result = processing_tasks.process_video_task.run(
                    str(video_path),
                    "en",
                    target_lang,
                    auto_create_video,
                    whisper_model,
                    translation_service="openai",
                    watermark_config=watermark_config,
                    processing_info=processing_info,
                    spotting_v2=spotting_v2,
                    translation_v2=translation_v2,
                    translation_style=translation_style,
                    render_v2=render_v2,
                )
            finally:
                processing_tasks.process_video_task.pop_request()
        return result, spy, str(downloads), saved_stats

    yield run


def payload(result):
    """Unwrap a successful task result.

    ``process_video_task`` returns ``{"status": "SUCCESS", "result": {...}}`` when it
    works and a flat ``{"status": "FAILURE", "error": ...}`` when it does not, so the
    two shapes have to be told apart before anything is read out of them.
    """
    assert result.get("status") == "SUCCESS", (
        f"job failed: {result.get('error', result)}"
    )
    return result["result"]


def read_srt(downloads, name):
    path = os.path.join(downloads, name)
    assert os.path.exists(path), f"{name} was never written (in {os.listdir(downloads)})"
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def srt_cue_texts(content):
    """The text lines of an SRT, in order — cue numbers and timings stripped."""
    texts = []
    for block in content.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            texts.append(" ".join(lines[2:]).strip())
        elif len(lines) == 2:  # number + timing, no text at all: a blank cue
            texts.append("")
    return texts


def assert_no_blank_cues(content, label):
    texts = srt_cue_texts(content)
    assert texts, f"{label}: no cues at all"
    blanks = [i for i, text in enumerate(texts, 1) if not text.strip()]
    assert not blanks, (
        f"{label}: cues {blanks} of {len(texts)} are blank — this is the "
        f"create_srt_file falsy-fallback defect\n{content[:400]}"
    )


#: The eight pipeline shapes: every combination of the three stage toggles.
COMBOS = [
    (spotting, translation, render)
    for spotting in (False, True)
    for translation in (False, True)
    for render in (False, True)
]
COMBO_IDS = [
    f"spot={int(s)}_trans={int(t)}_render={int(r)}" for s, t, r in COMBOS
]


# ==================================================================================
# 1. every combination, with and without a watermark
# ==================================================================================
@pytest.mark.unit
class TestAllFlagCombinations:
    @pytest.mark.parametrize("watermark", [False, True], ids=["no_wm", "watermark"])
    @pytest.mark.parametrize("spotting,translation,render", COMBOS, ids=COMBO_IDS)
    def test_job_succeeds_and_writes_usable_subtitles(
        self, pipeline, spotting, translation, render, watermark
    ):
        """Sixteen runs. Every one must produce two non-blank SRT files and a video."""
        result, spy, downloads, _stats = pipeline(
            spotting_v2=spotting,
            translation_v2=translation,
            render_v2=render,
            watermark=watermark,
        )

        files = payload(result)["files"]
        assert files["video_with_subtitles"], "no final video was produced"

        original = read_srt(downloads, files["original_srt"])
        translated = read_srt(downloads, files["translated_srt"])
        assert_no_blank_cues(original, "original_srt")
        assert_no_blank_cues(translated, "translated_srt")

        # The translated file must actually differ from the source one; the defect this
        # suite exists for produced a file that was *present*, *well-formed* and empty.
        assert srt_cue_texts(original) != srt_cue_texts(translated), (
            "translated SRT is identical to the source — no translation was applied"
        )

    @pytest.mark.parametrize("spotting,translation,render", COMBOS, ids=COMBO_IDS)
    def test_the_expected_stages_ran_and_the_others_did_not(
        self, pipeline, spotting, translation, render
    ):
        """Each toggle must select its own stage and leave the other branch alone."""
        _result, spy, _downloads, _stats = pipeline(
            spotting_v2=spotting, translation_v2=translation, render_v2=render
        )

        # --- transcription --------------------------------------------------------
        if spotting or translation:
            assert spy.ran("transcribe_with_words"), "v2 needs a separate transcription"
            assert not spy.ran("transcribe_and_translate_streamed")
            assert spy.calls["transcribe_with_words"][0]["collect_words"] is spotting, (
                "words are only collected when spotting_v2 asks for them"
            )
        else:
            assert spy.ran("transcribe_and_translate_streamed"), (
                "with both v2 stages off the legacy overlapped path must be used"
            )
            assert not spy.ran("transcribe_with_words")

        # --- translation ----------------------------------------------------------
        if translation:
            assert spy.ran("translate_cues") and spy.ran("enforce_cps")
            assert not spy.ran("translate_segments")
        elif spotting:
            # spotting_v2 alone still forces the split transcription, so translation
            # falls to the legacy per-segment translator.
            assert spy.ran("translate_segments")
            assert not spy.ran("translate_cues")
        else:
            assert not spy.ran("translate_cues") and not spy.ran("translate_segments")

        # --- rendering ------------------------------------------------------------
        if render:
            assert spy.ran("create_video_with_ass")
            assert not spy.ran("create_video_with_subtitles")
        else:
            assert spy.ran("create_video_with_subtitles")
            assert not spy.ran("create_video_with_ass")

    @pytest.mark.parametrize("render", [False, True], ids=["legacy_render", "render_v2"])
    def test_watermark_selects_the_combined_renderer(self, pipeline, render):
        """A watermark must not silently fall back to the subtitles-only renderer."""
        _result, spy, _downloads, _stats = pipeline(render_v2=render, watermark=True)

        if render:
            # render_v2 does subtitles+watermark in one ass/overlay graph.
            assert spy.ran("create_video_with_ass")
            assert spy.calls["create_video_with_ass"][0]["watermark_path"]
            assert not spy.ran("create_video_with_subtitles_and_watermark")
        else:
            assert spy.ran("create_video_with_subtitles_and_watermark")
            assert not spy.ran("create_video_with_subtitles")

    def test_spotting_v2_respots_from_words_not_whisper_segments(self, pipeline):
        """The point of spotting_v2: cue boundaries come from words, so they differ."""
        with_spotting, _spy, downloads_a, _s = pipeline(spotting_v2=True)
        without, _spy2, downloads_b, _s2 = pipeline(spotting_v2=False)

        respotted = srt_cue_texts(
            read_srt(downloads_a, payload(with_spotting)["files"]["original_srt"])
        )
        legacy = srt_cue_texts(
            read_srt(downloads_b, payload(without)["files"]["original_srt"])
        )
        # Same words, different cue boundaries — the merge/lead-out passes did something.
        assert respotted != legacy, "spotting_v2 produced Whisper's own segmentation"
        assert " ".join(respotted).split() == " ".join(legacy).split(), (
            "re-spotting changed the WORDS, not just the boundaries"
        )


# ==================================================================================
# 2. content, not just call counts
# ==================================================================================
@pytest.mark.unit
class TestSubtitleFileContent:
    def test_translation_v2_output_reaches_the_translated_srt(self, pipeline):
        """``translate_cues`` writes ``translated``; the SRT reads ``translated_text``.

        The hand-off between them is ``normalize_cues``. If it ever stops mapping the
        two spellings, this is where it shows up.
        """
        result, _spy, downloads, _stats = pipeline(spotting_v2=True, translation_v2=True)
        translated = read_srt(downloads, payload(result)["files"]["translated_srt"])
        assert "לסבך" in translated, f"the Hebrew never made it to disk:\n{translated}"
        assert "complicate" not in translated, "source text leaked into the translation"

    def test_source_srt_keeps_the_source_language(self, pipeline):
        result, _spy, downloads, _stats = pipeline(spotting_v2=True, translation_v2=True)
        original = read_srt(downloads, payload(result)["files"]["original_srt"])
        assert "complicate" in original
        assert "לסבך" not in original

    def test_spotting_without_translation_still_writes_real_text(self, pipeline):
        """The exact combination the falsy-fallback bug blanked out.

        ``spotting_v2`` on, ``translation_v2`` off, translation requested: cues arrive
        from ``normalize_cues`` carrying ``translated_text``, and if that value is falsy
        rather than absent the SRT writer must fall back to the source text.
        """
        result, _spy, downloads, _stats = pipeline(spotting_v2=True, translation_v2=False)
        translated = read_srt(downloads, payload(result)["files"]["translated_srt"])
        assert_no_blank_cues(translated, "translated_srt")
        assert "complicate" in translated or "legacy-he" in translated

    def test_no_translation_requested_writes_the_source_into_both_files(self, pipeline):
        """``target_lang='auto'`` means no translation — not an empty translated file."""
        result, _spy, downloads, _stats = pipeline(
            spotting_v2=True, target_lang="auto"
        )
        translated = read_srt(downloads, payload(result)["files"]["translated_srt"])
        assert_no_blank_cues(translated, "translated_srt")
        assert "complicate" in translated

    def test_srt_timings_are_well_formed_and_monotonic(self, pipeline):
        import re

        result, _spy, downloads, _stats = pipeline(spotting_v2=True, translation_v2=True)
        content = read_srt(downloads, payload(result)["files"]["translated_srt"])
        stamps = re.findall(
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", content
        )
        assert len(stamps) == len(srt_cue_texts(content))
        for start, end in stamps:
            assert start < end, f"non-positive duration: {start} --> {end}"
        for (_s1, end1), (start2, _e2) in zip(stamps, stamps[1:]):
            assert end1 <= start2, f"overlapping cues: {end1} then {start2}"

    def test_translation_style_is_forwarded_to_the_translator(self, pipeline):
        for style in (STYLE_CLEAN, STYLE_FAITHFUL):
            _result, spy, _downloads, _stats = pipeline(
                translation_v2=True, translation_style=style
            )
            assert spy.calls["translate_cues"][-1]["style"] == style

    def test_translation_progress_is_reported_during_the_v2_pass(self, pipeline):
        """Without a callback the UI sits on one frozen step for the whole translation."""
        _result, spy, _downloads, _stats = pipeline(translation_v2=True)
        assert spy.calls["translate_cues"][0]["has_progress_callback"]
        assert spy.calls["enforce_cps"][0]["has_progress_callback"]


# ==================================================================================
# 3. the flags and the numbers that come back out
# ==================================================================================
@pytest.mark.unit
class TestResultAndStatsReporting:
    @pytest.mark.parametrize("spotting,translation,render", COMBOS, ids=COMBO_IDS)
    def test_every_result_reports_the_flags_that_produced_it(
        self, pipeline, spotting, translation, render
    ):
        """An A/B experiment is unreadable if a result cannot be attributed to a setting."""
        result, _spy, _downloads, _stats = pipeline(
            spotting_v2=spotting, translation_v2=translation, render_v2=render
        )
        choices = payload(result)["user_choices"]
        assert choices["spotting_v2"] is spotting
        assert choices["translation_v2"] is translation
        assert choices["render_v2"] is render
        assert choices["translation_style"] == STYLE_CLEAN

    def test_route_supplied_user_choices_are_preserved_alongside_the_flags(self, pipeline):
        result, _spy, _downloads, _stats = pipeline(
            spotting_v2=True,
            processing_info={"user_choices": {"whisper_model": "tiny", "url": "u"}},
        )
        choices = payload(result)["user_choices"]
        assert choices["whisper_model"] == "tiny" and choices["url"] == "u"
        assert choices["spotting_v2"] is True

    def test_resolved_flags_are_reported_not_the_raw_arguments(self, pipeline):
        """Gemini has no word timestamps, so spotting_v2 is refused for that model.

        The result must say what actually ran, otherwise the experiment records a
        setting the job did not use.
        """
        result, spy, _downloads, _stats = pipeline(
            spotting_v2=True, translation_v2=True, whisper_model="gemini"
        )
        assert payload(result)["user_choices"]["spotting_v2"] is False
        assert spy.calls["transcribe_with_words"][0]["collect_words"] is False

    def test_translation_v2_records_real_token_usage(self, pipeline):
        """``translate_cues`` + ``enforce_cps`` usage is summed into the stats row."""
        _result, _spy, _downloads, stats = pipeline(translation_v2=True)
        row = stats[-1]
        assert row["translation_tokens"] == 1800, row  # 1500 + 300
        assert row["translation_cost_usd"] > 0
        assert row["translation_v2"] is True
        assert row["subtitle_pipeline_v2"] is True
        assert row["cues"] > 0

    def test_legacy_translation_records_zero_rather_than_an_invented_number(self, pipeline):
        """The legacy translators report no usage at all. Zero is the honest answer."""
        _result, _spy, _downloads, stats = pipeline(translation_v2=False)
        row = stats[-1]
        assert row["translation_tokens"] == 0
        assert row["translation_cost_usd"] == 0.0
        assert row["translation_v2"] is False

    def test_stats_row_carries_every_flag(self, pipeline):
        _result, _spy, _downloads, stats = pipeline(
            spotting_v2=True, render_v2=True, translation_style=STYLE_FAITHFUL
        )
        row = stats[-1]
        assert row["spotting_v2"] is True
        assert row["render_v2"] is True
        assert row["translation_style"] == STYLE_FAITHFUL
        assert row["subtitle_pipeline_v2"] is True

    def test_all_flags_off_is_marked_as_the_legacy_pipeline(self, pipeline):
        _result, _spy, _downloads, stats = pipeline()
        assert stats[-1]["subtitle_pipeline_v2"] is False


# ==================================================================================
# 4. failure handling
# ==================================================================================
@pytest.mark.unit
class TestTranslationFailureSalvage:
    """Translation failing must fail the job — but not throw the transcription away."""

    @pytest.fixture()
    def failed_run(self, pipeline):
        """One v2 run whose translation raises. Returns ``(result, downloads)``."""

        def boom(*_args, **_kwargs):
            raise RuntimeError("openai exploded")

        result, _spy, downloads, _stats = pipeline(
            translation_v2=True, translate_cues_impl=boom
        )
        return result, downloads

    def test_the_job_still_fails(self, failed_run):
        """The one thing this pipeline must never do is call untranslated output a success."""
        result, _downloads = failed_run
        assert result["status"] == "FAILURE"
        assert "openai exploded" in result["error"]

    def test_the_source_srt_survives_and_is_offered_back(self, failed_run):
        """Transcription is the expensive part; losing it to a translation outage is waste."""
        result, downloads = failed_run
        assert result.get("salvaged") is True
        content = read_srt(downloads, result["files"]["original_srt"])
        assert_no_blank_cues(content, "salvaged original_srt")
        assert "complicate" in content, "the salvaged file is not the transcription"

    def test_no_translated_srt_is_left_behind_pretending_to_be_a_translation(self, failed_run):
        """Source text under the translated filename is the failure mode to avoid."""
        result, downloads = failed_run
        assert "translated_srt" not in result.get("files", {})
        assert not os.path.exists(os.path.join(downloads, "clip_translated.srt")), (
            "a source-language file was written under the translated name"
        )


# ==================================================================================
# 5. proof that the content assertions above can actually fail
# ==================================================================================
@pytest.mark.unit
class TestTheContentAssertionsActuallyCatchTheRegression:
    """Revert the fix, show these tests go red, restore it.

    A regression test that has never been seen to fail is a guess. This reinstates the
    exact pre-fix line — ``if text is None`` instead of ``if not text`` — and asserts
    that the suite above catches it.
    """

    @staticmethod
    def _pre_fix_create_srt_file(segments, output_path, use_translation=False, language="en"):
        """``create_srt_file`` as it was before the fix, reduced to the deciding line."""
        from services.subtitle_service import format_timestamp

        with open(output_path, "w", encoding="utf-8") as handle:
            for index, segment in enumerate(segments, 1):
                text = (
                    segment.get("translated_text")
                    if use_translation
                    else segment.get("text", "")
                )
                if text is None:  # <-- THE BUG: "" is not None, so no fallback happened
                    text = segment.get("text", "")
                text = text.replace("\n", " ").replace("\r", " ")
                handle.write(
                    f"{index}\n{format_timestamp(segment['start'])} --> "
                    f"{format_timestamp(segment['end'])}\n{text}\n\n"
                )
        return output_path

    def test_reverting_the_fix_blanks_the_translated_srt(self, pipeline):
        """spotting_v2 on, translation_v2 off: the combination that shipped blank files."""
        from services import subtitle_service as subtitle_service_module

        service = subtitle_service_module.subtitle_service
        with patch.object(service, "create_srt_file", self._pre_fix_create_srt_file):
            result, _spy, downloads, _stats = pipeline(
                spotting_v2=True, translation_v2=False, target_lang="auto"
            )
            content = read_srt(downloads, payload(result)["files"]["translated_srt"])

        # The job "succeeded" — that is precisely what made the bug survive to production.
        assert result.get("status") != "FAILURE"
        with pytest.raises(AssertionError, match="are blank"):
            assert_no_blank_cues(content, "translated_srt")

    def test_with_the_fix_in_place_the_same_run_is_clean(self, pipeline):
        """The other half of the proof: same inputs, real code, no blank cues."""
        result, _spy, downloads, _stats = pipeline(
            spotting_v2=True, translation_v2=False, target_lang="auto"
        )
        assert_no_blank_cues(
            read_srt(downloads, payload(result)["files"]["translated_srt"]), "translated_srt"
        )

    def test_the_blank_cue_detector_is_not_vacuous(self):
        """``assert_no_blank_cues`` must reject a file of empty cues and accept a real one."""
        blank = "1\n00:00:00,000 --> 00:00:01,000\n\n\n2\n00:00:01,000 --> 00:00:02,000\n\n\n"
        with pytest.raises(AssertionError):
            assert_no_blank_cues(blank, "synthetic")
        assert_no_blank_cues(
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n", "synthetic"
        )
