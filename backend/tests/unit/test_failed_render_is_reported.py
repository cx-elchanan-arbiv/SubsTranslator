"""
What ``process_video_task`` reports when a stage does NOT succeed.

Why this file exists
--------------------
Every defect below was observed on a real run, and every one of them looked like a
finished job to the person who submitted it:

* the burn-in failed and the UI showed "Embedding Subtitles — completed 100%",
  "Finalizing Video — completed 100%", 100% overall, not one step in an error state,
  and a download button that simply was not there. No error appeared anywhere.
* the stats row written to Redis for that run said ``"status": "success"`` with a null
  error, because the literal was hardcoded.
* the legacy translator answered a failure by copying the SOURCE text into
  ``translated_text``, which ships an untranslated .srt under a green tick.
* the stats row recorded ``transcription_model`` = the model the caller ASKED for. It
  is the model that RAN that the corpus needs, and whisper_smart downgrades on low
  memory, on a load failure, on a transcribe exception and on a Gemini fallback.

The tests are therefore assertions about HONESTY, not about rendering: no ffmpeg runs
here. The stage boundaries are mocked and made to fail; everything between them — the
progress bookkeeping, the SRT writing, the stats payload and the returned result — is
the real code.
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

WHISPER_SEGMENTS = [
    {"start": 0.0, "end": 1.3, "text": "It could complicate"},
    {"start": 1.3, "end": 3.8, "text": "things. Do you worry about that?"},
    {"start": 4.2, "end": 6.0, "text": "Yeah, I think about it."},
]

STREAMED_SEGMENTS = [
    {**segment, "translated_text": f"[legacy-he] {segment['text']}"}
    for segment in WHISPER_SEGMENTS
]

STEP_EMBED = 5
STEP_FINALIZE = 6


class RecorderSpy:
    """Inert stand-in for the research recorder (and keeps the corpus clean)."""

    active = False
    run_dir = "/fake/research/run"

    def __init__(self, task_id):
        self.task_id = task_id
        self.meta = {}
        self.finished = None

    def update_meta(self, **fields):
        self.meta.update(fields)

    def finish(self, success, error=None):
        self.finished = {"success": success, "error": error}

    def __getattr__(self, name):
        return lambda *_a, **_k: None


class Run:
    """One finished job: what it returned, what it wrote, what the UI was told."""

    def __init__(self, result, downloads, stats, recorder, steps):
        self.result = result
        self.downloads = downloads
        self.stats = stats
        self.recorder = recorder
        #: The step list exactly as it was last pushed to Celery — i.e. the payload
        #: /status/<task_id> serves to the progress display.
        self.steps = steps

    def path(self, filename):
        return os.path.join(self.downloads, filename)


def _last_reported_steps(update_state_mock):
    for call in reversed(update_state_mock.call_args_list):
        meta = call.kwargs.get("meta")
        if isinstance(meta, dict) and "steps" in meta:
            return meta["steps"]
    return []


@pytest.fixture()
def run_job(tmp_path):
    """Run ``process_video_task`` with the stage boundaries under the test's control.

    Yields ``run(**kwargs) -> Run``. Renderers default to succeeding; a test overrides
    exactly the one it wants to break.
    """
    counter = {"n": 0}
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 1024)
    logo_path = tmp_path / "logo.png"
    logo_path.write_bytes(b"\x89PNG" + b"\x00" * 64)

    def _touch(path):
        with open(path, "wb") as handle:
            handle.write(b"\x00" * 2048)

    def run(
        *,
        renderer=None,
        combined_renderer=None,
        transcribe_streamed=None,
        translate_segments_impl=None,
        watermark=False,
        spotting_v2=False,
        translation_v2=False,
        # Pinned explicitly, all three. These tests are about what the task REPORTS
        # when a stage fails, not about which pipeline runs — so the pipeline must
        # not move under them when the product defaults change.
        render_v2=False,
        target_lang="he",
        whisper_model="large",
        last_model_used=None,
        transcription_model_used=None,
    ):
        from services import subtitle_service as subtitle_service_module
        from tasks import processing_tasks

        counter["n"] += 1
        downloads = tmp_path / f"downloads_{counter['n']}"
        downloads.mkdir()

        recorder_box = {}

        def fake_start_run(task_id, **_kwargs):
            recorder_box["recorder"] = RecorderSpy(task_id)
            return recorder_box["recorder"]

        def fake_transcribe_with_words(path, **kwargs):
            result = {
                "language": "en",
                "segments": [dict(s) for s in WHISPER_SEGMENTS],
                "words": [],
            }
            if transcription_model_used:
                result["model_used"] = transcription_model_used
            return result

        def default_streamed(path, **kwargs):
            result = {
                "language": "en",
                "segments": [dict(s) for s in STREAMED_SEGMENTS],
            }
            if transcription_model_used:
                result["model_used"] = transcription_model_used
            return result

        def fake_translate_segments(segments, target, **kwargs):
            return [
                {**segment, "translated_text": f"[legacy-he] {segment['text']}"}
                for segment in segments
            ]

        def default_renderer(video, srt, output, language, **kwargs):
            _touch(output)
            return True

        def default_combined(video, srt, output, watermark_path, language, **kwargs):
            _touch(output)
            return True

        stats = []
        service = subtitle_service_module.subtitle_service
        watermark_config = (
            {"enabled": True, "custom_logo_path": str(logo_path)} if watermark else None
        )

        # The task reads `smart_whisper.last_model_used` to find out which model really
        # ran; nothing loads a model in this test, so stand in for the manager.
        model_manager = MagicMock()
        model_manager.last_model_used = last_model_used
        update_state = MagicMock()

        with (
            patch.multiple(
                processing_tasks,
                DOWNLOADS_FOLDER=str(downloads),
                transcribe_with_words=fake_transcribe_with_words,
                transcribe_and_translate_streamed=transcribe_streamed
                or default_streamed,
                transcribe_video=lambda path, **kw: {
                    "language": "en",
                    "segments": [dict(s) for s in WHISPER_SEGMENTS],
                },
                translate_segments=translate_segments_impl or fake_translate_segments,
                save_video_stats=stats.append,
                start_run=fake_start_run,
                smart_whisper=model_manager,
            ),
            patch.object(processing_tasks.time, "sleep", lambda *_a, **_k: None),
            patch.object(
                service, "create_video_with_subtitles", renderer or default_renderer
            ),
            patch.object(
                service,
                "create_video_with_subtitles_and_watermark",
                combined_renderer or default_combined,
            ),
            patch.object(
                processing_tasks.process_video_task, "update_state", update_state
            ),
        ):
            processing_tasks.process_video_task.push_request(id="test-task-000000")
            try:
                result = processing_tasks.process_video_task.run(
                    str(video_path),
                    "en",
                    target_lang,
                    True,  # auto_create_video
                    whisper_model,
                    translation_service="google",
                    watermark_config=watermark_config,
                    spotting_v2=spotting_v2,
                    render_v2=render_v2,
                    translation_v2=translation_v2,
                )
            finally:
                processing_tasks.process_video_task.pop_request()

        return Run(
            result,
            str(downloads),
            stats,
            recorder_box.get("recorder"),
            _last_reported_steps(update_state),
        )

    yield run


# ==================================================================================
# 1. a failed burn is a failed job
# ==================================================================================
@pytest.mark.unit
class TestAFailedBurnIsReportedAsAFailure:
    def _assert_reported_as_failure(self, result):
        assert result["status"] == "FAILURE", f"reported as a success: {result}"
        assert result["code"] == "RENDER_FAILED"
        # /status/<task_id> hands `user_facing_message` straight to the UI. Without it
        # the user gets the generic "Processing failed" and no idea the .srt survived.
        assert result["user_facing_message"]
        assert ".srt" in result["user_facing_message"]
        assert result["error"]

    def test_a_failed_burn_without_watermark_fails_the_job(self, run_job):
        run = run_job(renderer=lambda *a, **k: False)

        self._assert_reported_as_failure(run.result)

        # The artefacts that DO exist are handed over, not discarded.
        files = run.result["files"]
        assert files["video_with_subtitles"] is None
        assert os.path.exists(run.path(files["original_srt"]))
        assert os.path.exists(run.path(files["translated_srt"]))
        assert run.result["salvaged"] is True

        # ...and the run is archived as a failure, not as a win.
        assert run.recorder.finished["success"] is False

    def test_a_failed_burn_with_a_watermark_fails_the_job(self, run_job):
        run = run_job(watermark=True, combined_renderer=lambda *a, **k: False)

        self._assert_reported_as_failure(run.result)
        assert run.result["files"]["video_with_subtitles"] is None
        assert os.path.exists(run.path(run.result["files"]["original_srt"]))

    def test_a_renderer_that_reports_success_but_writes_nothing_is_caught(
        self, run_job
    ):
        """The bool is FFmpeg's exit status; the download link is the file itself."""

        def lying_renderer(video, srt, output, language, **kwargs):
            return True  # ...and writes no file at all

        run = run_job(renderer=lying_renderer)

        self._assert_reported_as_failure(run.result)

    @pytest.mark.parametrize("watermark", [False, True], ids=["no_wm", "watermark"])
    def test_the_progress_steps_show_the_failure_instead_of_100_percent(
        self, run_job, watermark
    ):
        run = run_job(
            watermark=watermark,
            renderer=lambda *a, **k: False,
            combined_renderer=lambda *a, **k: False,
        )

        embed, finalize = run.steps[STEP_EMBED], run.steps[STEP_FINALIZE]
        assert embed["status"] == "error", (
            f"'Embedding Subtitles' is {embed['status']!r} after a failed burn — this "
            f"is the 'completed 100%' the user was shown"
        )
        assert finalize["status"] != "completed", (
            f"'Finalizing Video' is {finalize['status']!r} after a burn that never "
            f"produced a video"
        )
        assert any(step["status"] == "error" for step in run.steps), (
            "no step is in an error state, so the progress display has nothing red to "
            "show and the job looks finished"
        )

    def test_a_successful_burn_still_reports_success(self, run_job):
        """Control: the honest path must not have been broken by any of the above."""
        run = run_job()

        assert run.result["status"] == "SUCCESS"
        assert run.result["result"]["files"]["video_with_subtitles"]
        assert run.stats[0]["status"] == "success"
        assert run.stats[0]["error_message"] is None
        assert run.recorder.finished["success"] is True

        assert run.steps[STEP_EMBED]["status"] == "completed"
        assert run.steps[STEP_FINALIZE]["status"] == "completed"


# ==================================================================================
# 2. the stats row says what happened
# ==================================================================================
@pytest.mark.unit
class TestTheStatsRowIsHonest:
    def test_a_failed_burn_is_recorded_as_a_failure(self, run_job):
        run = run_job(renderer=lambda *a, **k: False)

        assert run.stats, "no stats row was written for the failed run"
        row = run.stats[0]
        assert row["status"] == "failure", (
            f"stats recorded {row['status']!r} for a run that produced no video — "
            f"this field used to be the hardcoded literal 'success'"
        )
        assert row[
            "error_message"
        ], "a failed row with no error message explains nothing"

    def test_the_recorded_model_is_the_one_that_ran(self, run_job):
        """`large` was requested; whisper_smart downgraded to `medium` mid-run."""
        run = run_job(whisper_model="large", last_model_used="medium")

        assert run.stats[0]["transcription_model"] == "medium", (
            "the stats row (and the stats:index:model:<name> set built from it) "
            "recorded the REQUESTED model, not the one that actually transcribed"
        )
        assert run.stats[0]["transcription_model_requested"] == "large"
        assert run.recorder.meta["transcription_model_used"] == "medium"

    def test_the_transcribers_own_report_wins(self, run_job):
        run = run_job(
            whisper_model="large",
            last_model_used="medium",
            transcription_model_used="base",
        )

        assert run.stats[0]["transcription_model"] == "base"

    def test_no_downgrade_records_the_requested_model(self, run_job):
        run = run_job(whisper_model="large", last_model_used="large")

        assert run.stats[0]["transcription_model"] == "large"
        assert run.stats[0]["transcription_model_requested"] == "large"


# ==================================================================================
# 3. a translation that did not happen is a failure, with the transcript salvaged
# ==================================================================================
@pytest.mark.unit
class TestLegacyTranslationFailuresAreReported:
    def test_the_legacy_translator_failing_fails_the_job_and_keeps_the_transcript(
        self, run_job
    ):
        """spotting_v2 on + translation_v2 off = the legacy translator on the v2 path."""

        def exploding_translate_segments(segments, target, **kwargs):
            raise RuntimeError("google translate returned 503")

        run = run_job(
            spotting_v2=True,
            translate_segments_impl=exploding_translate_segments,
        )

        assert run.result["status"] == "FAILURE"
        assert run.result["salvaged"] is True
        salvaged = run.path(run.result["files"]["original_srt"])
        assert os.path.exists(salvaged), "the transcript was thrown away"
        with open(salvaged, encoding="utf-8") as handle:
            content = handle.read()
        assert "complicate" in content, "the salvaged file is not the transcription"
        assert run.recorder.finished["success"] is False

    def test_the_streamed_legacy_path_writes_the_salvaged_transcript(self, run_job):
        """The overlapped path has no SRT on disk when translation dies — so write one.

        Defensive on purpose: the raise lives in transcription_service and may hand over
        a filename it wrote itself, or the segments, or neither.
        """
        from tasks.processing_tasks import TranslationFailedWithSalvage

        def exploding_streamed(path, **kwargs):
            error = TranslationFailedWithSalvage("openai 429 after 3 retries", "")
            error.segments = [dict(s) for s in WHISPER_SEGMENTS]
            raise error

        run = run_job(transcribe_streamed=exploding_streamed)

        assert run.result["status"] == "FAILURE"
        assert "429" in run.result["error"]
        assert run.result["salvaged"] is True
        salvaged = run.path(run.result["files"]["original_srt"])
        assert os.path.exists(salvaged)
        with open(salvaged, encoding="utf-8") as handle:
            assert "complicate" in handle.read()

    def test_a_failure_with_nothing_to_salvage_does_not_advertise_a_file(self, run_job):
        """Naming a file that was never written sends the UI to a download that 404s."""
        from tasks.processing_tasks import TranslationFailedWithSalvage

        def exploding_streamed(path, **kwargs):
            raise TranslationFailedWithSalvage("translator unreachable", "")

        run = run_job(transcribe_streamed=exploding_streamed)

        assert run.result["status"] == "FAILURE"
        assert "files" not in run.result
        assert "salvaged" not in run.result

    def test_a_working_legacy_translation_still_succeeds(self, run_job):
        """Control: the guard must not turn a working translation into a failure."""
        run = run_job(spotting_v2=True)

        assert run.result["status"] == "SUCCESS"
        translated = run.path(run.result["result"]["files"]["translated_srt"])
        with open(translated, encoding="utf-8") as handle:
            assert "[legacy-he]" in handle.read()


# ==================================================================================
# 5. structured failures keep their story; translation gaps are counted
# ==================================================================================


class TestStructuredFailuresKeepTheirStory:
    """The two decisions from the bug review, plus the pipe that carries them.

    #1 (decision: FAILURE): zero transcription segments used to sail through as a
    green job with two 0-byte SRT files. Now it is a classified failure.

    #2 (decision: WARNING): a cue written from source text because its translation
    is missing no longer hides — the job succeeds and carries the count.

    #4: a ``VideoProcessingError`` reaching the task boundary keeps its code,
    user-facing message and recoverability instead of collapsing into a bare
    "Task failed".
    """

    def test_empty_transcription_is_a_classified_failure(self, run_job):
        run = run_job(
            transcribe_streamed=lambda path, **kw: {"language": "en", "segments": []},
        )
        assert run.result["status"] == "FAILURE"
        assert run.result["code"] == "TRANSCRIPTION_EMPTY"
        assert run.result["recoverable"] is False
        assert "speech" in run.result["user_facing_message"].lower()
        # No 0-byte files are offered with the failure.
        assert "files" not in run.result

    def test_a_structured_error_keeps_code_message_and_recoverability(self, run_job):
        from core.exceptions import VideoProcessingError

        def exploding(path, **kw):
            raise VideoProcessingError(
                message="disk exploded mid-write",
                error_code="FILE_PERMISSION_ERROR",
                recoverable=False,
                user_message="Permission denied to write the file.",
            )

        run = run_job(transcribe_streamed=exploding)
        assert run.result["status"] == "FAILURE"
        assert run.result["code"] == "FILE_PERMISSION_ERROR"
        assert (
            run.result["user_facing_message"] == "Permission denied to write the file."
        )
        assert run.result["recoverable"] is False
        # The raw message is still there; the traceback goes to the logs, where
        # log_task_error already writes it.
        assert "disk exploded" in run.result["error"]

    def test_untranslated_cues_are_counted_and_reported(self, run_job):
        def leaky(path, **kw):
            return {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "one", "translated_text": "אחת"},
                    {"start": 1.0, "end": 2.0, "text": "two"},  # missing entirely
                    {"start": 2.0, "end": 3.0, "text": "three", "translated_text": ""},
                ],
            }

        run = run_job(transcribe_streamed=leaky)
        assert run.result["status"] == "SUCCESS"
        assert run.result["result"]["translation_gaps"] == 2
        # The holes were filled with SOURCE text (the file is usable), not dropped.
        # Output names carry a slice of the task id (bug #9 fix) — the fixture
        # pins the id to "test-task-000000", so the slice is "test-tas".
        with open(run.path("clip_test-tas_translated.srt"), encoding="utf-8") as handle:
            translated = handle.read()
        assert "two" in translated
        assert "three" in translated
        assert "אחת" in translated

    def test_a_fully_translated_job_reports_no_gaps(self, run_job):
        run = run_job()
        assert run.result["status"] == "SUCCESS"
        # Clean payload: the key appears only when there is something to say.
        assert "translation_gaps" not in run.result["result"]


# ==================================================================================
# 6. the recorded translation provider is the one that RAN (bug #13)
# ==================================================================================


class TestTranslationProviderHonesty:
    """ "google" was chosen, OpenAI billed, and "google" recorded — because the
    v2 quality path translates with OpenAI regardless of the selector, and the
    stats echoed the REQUEST. Same correction the transcription model already
    got: record what ran, keep what was asked as its own field."""

    def test_legacy_path_records_the_requested_service_because_it_honors_it(
        self, run_job
    ):
        run = run_job()  # default: legacy P1, translation_service="google"
        assert run.result["status"] == "SUCCESS"
        assert run.result["result"]["translation_service_used"] == "google"
        stats = run.stats[0]
        assert stats["translation_service"] == "google"
        assert stats["translation_service_requested"] == "google"

    def test_v2_path_records_openai_even_when_google_was_requested(self, run_job):
        from tasks import processing_tasks

        class FakeUsage:
            total_tokens = 42
            cost_usd = 0.001

            def as_dict(self):
                return {"total_tokens": self.total_tokens, "cost_usd": self.cost_usd}

        class FakeTranslated(list):
            """translate_cues/enforce_cps contract: an iterable of cues that
            also carries .usage (and optionally .mode)."""

            usage = FakeUsage()
            mode = "translate"

        def fake_translate_cues(cues, target, **kwargs):
            return FakeTranslated(
                {**cue, "translated_text": f"[he] {cue.get('text', '')}"}
                for cue in cues
            )

        def fake_enforce_cps(translated, **kwargs):
            return FakeTranslated(dict(cue) for cue in translated)

        with patch.multiple(
            processing_tasks,
            translate_cues=fake_translate_cues,
            enforce_cps=fake_enforce_cps,
        ):
            run = run_job(spotting_v2=True, translation_v2=True)

        assert run.result["status"] == "SUCCESS"
        # The user asked for google; OpenAI is what the v2 path actually runs.
        assert run.result["result"]["translation_service_used"] == "openai"
        stats = run.stats[0]
        assert stats["translation_service"] == "openai"
        assert stats["translation_service_requested"] == "google"


class TestUnsupportedLanguageClassification:
    """Bug #17: an unsupported target language was classified as a generic
    transient failure and told the user to "try again" — advice that cannot
    work, since the same language pair is rejected every time."""

    def test_unsupported_language_gets_its_own_code(self):
        from tasks.processing_tasks import classify_translation_failure

        code, message = classify_translation_failure(
            "TranslationServiceError: google: No support for the provided language he-XX"
        )
        assert code == "TRANSLATION_UNSUPPORTED_LANGUAGE"
        assert "different target language" in message

    def test_quota_still_wins_over_everything(self):
        """Regression guard for the existing ordering: an out-of-credits reply
        is ALSO a 429, and must keep classifying as credits, not rate limit."""
        from tasks.processing_tasks import classify_translation_failure

        code, _ = classify_translation_failure(
            "Error code: 429 - insufficient_quota: no credits remaining"
        )
        assert code == "TRANSLATION_NO_CREDITS"
