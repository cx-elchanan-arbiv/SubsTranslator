"""
Unit tests for :mod:`services.research_recorder`.

Two independent obligations, and the second one matters more than the first:

1. **It archives the right things.** Directory naming that cannot collide, meta.json
   with the flags and the geometry, the cue list at each stage, every LLM exchange in
   call order, copies of the outputs, one index line per run.
2. **It can never take a job down.** Every method is asserted to swallow its own
   failures. The tests do not stub the recorder's error paths with polite fakes — they
   make real calls fail (unwritable directories, patched ``open``/``makedirs``/
   ``copy2`` raising ``OSError``, an un-serialisable payload, an exploding response
   object) and assert the recorder returns normally anyway. A recorder that raises is
   worse than no recorder, because it converts an observability bug into a lost job.
"""

import json
import os
import stat
import sys
from unittest.mock import patch

import pytest


def _find_backend_dir():
    for seed in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = seed
        while True:
            if os.path.isfile(os.path.join(path, "services", "research_recorder.py")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise RuntimeError("could not locate the backend directory containing services/")


backend_dir = _find_backend_dir()
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services import research_recorder as rr  # noqa: E402


@pytest.fixture()
def root(tmp_path):
    path = tmp_path / "research"
    path.mkdir()
    return str(path)


@pytest.fixture()
def recorder(root):
    return rr.start_run("abcdef1234567890", root=root, enabled=True)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class FakeResponse:
    """Stands in for an ``openai`` completion object (pydantic ``model_dump``)."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


# ======================================================================================
@pytest.mark.unit
class TestRunDirectory:
    def test_directory_is_timestamped_and_task_scoped(self, recorder, root):
        name = os.path.basename(recorder.run_dir)
        stamp, _, suffix = name.rpartition("_")
        assert suffix == "abcdef12"  # task_id[:8]
        assert len(stamp) == 15 and stamp[8] == "_"  # yyyymmdd_HHMMSS
        assert os.path.isdir(os.path.join(recorder.run_dir, "llm"))
        assert os.path.isdir(os.path.join(recorder.run_dir, "outputs"))

    def test_two_runs_of_the_same_video_do_not_collide(self, root):
        """The whole point: /app/downloads is keyed by filename, this is not."""
        first = rr.start_run("aaaaaaaa1111", root=root, enabled=True)
        second = rr.start_run("bbbbbbbb2222", root=root, enabled=True)
        assert first.run_dir != second.run_dir
        for rec in (first, second):
            rec.save_segments([{"start": 0, "end": 1, "text": "x"}])
            rec.finish(success=True)
        assert len(os.listdir(root)) == 3  # two run dirs + index.jsonl

    def test_active_flag_is_true_for_a_working_recorder(self, recorder):
        assert recorder.active is True


# ======================================================================================
@pytest.mark.unit
class TestArchivedContent:
    def test_meta_carries_the_experiment_variables(self, recorder):
        recorder.update_meta(
            spotting_v2=True,
            translation_v2=True,
            translation_style="clean",
            render_v2=True,
            whisper_model="large",
            translation_service="openai",
            pipeline="v2",
            detected_language="en",
        )
        recorder.add_timing("transcribe", 12.3456)
        recorder.add_timing("translate", 4.0)
        recorder.finish(success=True)

        meta = read_json(os.path.join(recorder.run_dir, "meta.json"))
        assert meta["task_id"] == "abcdef1234567890"
        assert meta["spotting_v2"] is True
        assert meta["translation_style"] == "clean"
        assert meta["whisper_model"] == "large"
        assert meta["pipeline"] == "v2"
        assert meta["timings"] == {"transcribe": 12.346, "translate": 4.0}
        assert meta["success"] is True
        assert meta["error"] is None
        assert meta["started_at"] and meta["finished_at"]
        assert isinstance(meta["wall_seconds"], float)

    def test_failure_is_archived_with_its_error(self, recorder):
        recorder.finish(success=False, error="TranslationV2Error: missing ids [3, 4]")
        meta = read_json(os.path.join(recorder.run_dir, "meta.json"))
        assert meta["success"] is False
        assert "missing ids" in meta["error"]

    def test_segments_words_and_cues(self, recorder):
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        words = [{"s": 0.0, "e": 0.4, "w": "hello"}]
        recorder.save_segments(segments)
        recorder.save_words(words)
        recorder.save_cues("pre_translation", segments)
        recorder.save_cues(
            "post_translation", [{**segments[0], "translated_text": "שלום"}]
        )
        recorder.save_cues("post_reflow", [{**segments[0], "translated_text": "שלום"}])
        recorder.finish(success=True)

        assert read_json(os.path.join(recorder.run_dir, "segments.json")) == segments
        assert read_json(os.path.join(recorder.run_dir, "words.json")) == words
        post = read_json(os.path.join(recorder.run_dir, "cues_post_translation.json"))
        assert post[0]["translated_text"] == "שלום"
        meta = read_json(os.path.join(recorder.run_dir, "meta.json"))
        assert meta["segments_count"] == 1
        assert meta["words_count"] == 1
        assert meta["cue_counts"] == {
            "pre_translation": 1,
            "post_translation": 1,
            "post_reflow": 1,
        }

    def test_hebrew_is_stored_as_text_not_escapes(self, recorder):
        """A corpus you have to un-escape by hand is a corpus nobody reads."""
        recorder.save_segments([{"text": "צה״ל הודיע"}])
        raw = open(
            os.path.join(recorder.run_dir, "segments.json"), encoding="utf-8"
        ).read()
        assert "צה״ל" in raw

    def test_empty_word_list_writes_no_file(self, recorder):
        """Gemini returns no word timings; an empty words.json would be a lie."""
        recorder.save_words([])
        assert not os.path.exists(os.path.join(recorder.run_dir, "words.json"))


# ======================================================================================
@pytest.mark.unit
class TestLlmCapture:
    def test_requests_are_numbered_in_call_order(self, recorder):
        for index in range(3):
            recorder.record_llm(
                stage=f"translate_chunk_{index + 1}",
                system=f"system {index}",
                user=f"user {index}",
                response=FakeResponse({"id": f"chatcmpl-{index}"}),
                meta={"model": "gpt-4o", "latency_s": 1.5, "prompt_tokens": 100},
            )
        llm_dir = os.path.join(recorder.run_dir, "llm")
        assert sorted(os.listdir(llm_dir)) == sorted(
            f"{n:02d}_{suffix}"
            for n in (1, 2, 3)
            for suffix in ("system.txt", "user.txt", "response.json", "meta.json")
        )
        assert (
            open(os.path.join(llm_dir, "02_system.txt"), encoding="utf-8").read()
            == "system 1"
        )
        assert (
            read_json(os.path.join(llm_dir, "02_response.json"))["id"] == "chatcmpl-1"
        )
        meta = read_json(os.path.join(llm_dir, "03_meta.json"))
        assert meta["stage"] == "translate_chunk_3"
        assert meta["index"] == 3
        assert meta["model"] == "gpt-4o"
        assert meta["prompt_tokens"] == 100

    def test_a_failed_request_is_still_recorded(self, recorder):
        recorder.record_llm(
            stage="translate_chunk_1",
            system="s",
            user="u",
            response=None,
            meta={"model": "gpt-4o", "error": "APITimeoutError: timed out"},
        )
        llm_dir = os.path.join(recorder.run_dir, "llm")
        assert not os.path.exists(os.path.join(llm_dir, "01_response.json"))
        assert "timed out" in read_json(os.path.join(llm_dir, "01_meta.json"))["error"]

    def test_a_response_object_without_model_dump_still_lands(self, recorder):
        class Odd:
            def model_dump(self):
                raise RuntimeError("not pydantic")

            def to_dict(self):
                raise RuntimeError("not that either")

            def __repr__(self):
                return "<odd response>"

        recorder.record_llm(stage="s", system="s", user="u", response=Odd(), meta={})
        payload = read_json(os.path.join(recorder.run_dir, "llm", "01_response.json"))
        assert payload == {"repr": "<odd response>"}

    def test_request_count_reaches_meta(self, recorder):
        for _ in range(4):
            recorder.record_llm(stage="s", system="s", user="u", response=None, meta={})
        recorder.finish(success=True)
        assert (
            read_json(os.path.join(recorder.run_dir, "meta.json"))["llm_requests"] == 4
        )


# ======================================================================================
@pytest.mark.unit
class TestOutputs:
    def test_files_are_copied_with_their_sizes_recorded(self, recorder, tmp_path):
        srt = tmp_path / "clip_translated.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nשלום\n", encoding="utf-8")
        recorder.copy_output(str(srt))
        recorder.finish(success=True)

        copied = os.path.join(recorder.run_dir, "outputs", "clip_translated.srt")
        assert os.path.exists(copied)
        assert "שלום" in open(copied, encoding="utf-8").read()
        assert (
            "clip_translated.srt"
            in read_json(os.path.join(recorder.run_dir, "meta.json"))["outputs"]
        )

    def test_a_missing_file_is_a_no_op(self, recorder):
        recorder.copy_output("/nonexistent/clip_final.mp4")
        assert os.listdir(os.path.join(recorder.run_dir, "outputs")) == []

    def test_oversized_files_are_skipped_and_the_skip_is_recorded(self, root, tmp_path):
        # A non-video extension on purpose: video never reaches the size check at
        # all (see the test below), so the oversize guard is exercised on the kind
        # of file it still applies to.
        rec = rr.ResearchRecorder("cafebabe0000", root, max_copy_mb=0)
        big = tmp_path / "huge.wav"
        big.write_bytes(b"\x00" * (2 * 1024 * 1024))
        rec.copy_output(str(big))
        rec.finish(success=True)
        assert os.listdir(os.path.join(rec.run_dir, "outputs")) == []
        warnings = read_json(os.path.join(rec.run_dir, "meta.json"))[
            "recorder_warnings"
        ]
        assert any("huge.wav" in w for w in warnings)

    def test_video_is_referenced_never_copied(self, recorder, tmp_path):
        """The archive answers "what text did the pipeline produce and why";
        duplicating every rendered MP4 multiplied a run's storage ~100x for a
        file re-renderable from the archived cues. The reference in meta.json
        keeps the archive honest about what the run produced.
        """
        clip = tmp_path / "clip_final.mp4"
        clip.write_bytes(b"\x00" * 1024)
        recorder.copy_output(str(clip))
        recorder.finish(success=True)

        assert os.listdir(os.path.join(recorder.run_dir, "outputs")) == []
        meta = read_json(os.path.join(recorder.run_dir, "meta.json"))
        entry = meta["outputs"]["clip_final.mp4"]
        assert entry["archived"] is False
        assert entry["path"] == str(clip)
        assert entry["size_mb"] == 0.0


# ======================================================================================
@pytest.mark.unit
class TestIndex:
    def test_one_line_per_run(self, root):
        for index in range(3):
            rec = rr.start_run(f"task{index}0000000", root=root, enabled=True)
            rec.update_meta(pipeline="v2", spotting_v2=True, whisper_model="large")
            rec.finish(success=True)
        lines = (
            open(os.path.join(root, "index.jsonl"), encoding="utf-8")
            .read()
            .splitlines()
        )
        assert len(lines) == 3
        rows = [json.loads(line) for line in lines]
        assert [row["task_id"] for row in rows] == [f"task{i}0000000" for i in range(3)]
        assert all(row["pipeline"] == "v2" and row["success"] for row in rows)
        assert all(row["dir"] for row in rows)

    def test_index_row_carries_the_scan_keys(self, recorder, root):
        recorder.update_meta(
            pipeline="v2",
            spotting_v2=True,
            translation_v2=True,
            translation_style="faithful",
            render_v2=True,
            whisper_model="large",
            translation_service="openai",
            source_name="IMG_8975.MP4",
            video_width=720,
            video_height=1280,
            translation_tokens=12345,
            translation_cost_usd=0.0421,
        )
        recorder.save_cues("post_reflow", [{}, {}, {}])
        recorder.finish(success=True)
        row = json.loads(
            open(os.path.join(root, "index.jsonl"), encoding="utf-8").read()
        )
        assert row["video"] == "IMG_8975.MP4"
        assert (row["w"], row["h"]) == (720, 1280)
        assert row["cues"] == 3
        assert row["cost_usd"] == 0.0421
        assert row["translation_style"] == "faithful"


# ======================================================================================
@pytest.mark.unit
class TestItCanNeverFailTheJob:
    """The safety contract. Every one of these must return normally."""

    def test_an_uncreatable_root_yields_an_inert_recorder(self, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        os.chmod(blocked, stat.S_IRUSR | stat.S_IXUSR)  # r-x: no writing inside
        try:
            rec = rr.start_run(
                "deadbeef0000", root=str(blocked / "research"), enabled=True
            )
            assert rec.active is False
            # ...and every call on it is a silent no-op, not an AttributeError.
            rec.update_meta(pipeline="v2")
            rec.record_source_video("/whatever.mp4")
            rec.save_segments([{"text": "x"}])
            rec.save_words([{"w": "x"}])
            rec.save_cues("pre_translation", [{}])
            rec.record_llm(stage="s", system="s", user="u", response=None, meta={})
            rec.copy_output(__file__)
            rec.add_timing("transcribe", 1.0)
            rec.finish(success=True)
        finally:
            os.chmod(blocked, stat.S_IRWXU)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_read_only_run_directory_does_not_raise(self, recorder):
        os.chmod(recorder.run_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            recorder.save_segments([{"text": "x"}])
            recorder.save_cues("pre_translation", [{}])
            recorder.record_llm(stage="s", system="s", user="u", response=None, meta={})
            recorder.copy_output(__file__)
            recorder.finish(success=True)
        finally:
            os.chmod(recorder.run_dir, stat.S_IRWXU)

    @pytest.mark.parametrize(
        ("method", "args", "kwargs"),
        [
            ("save_segments", ([{"text": "x"}],), {}),
            ("save_words", ([{"w": "x"}],), {}),
            ("save_cues", ("pre_translation", [{}]), {}),
            ("record_llm", (), {"stage": "s", "system": "s", "user": "u", "meta": {}}),
            ("finish", (), {"success": True}),
        ],
    )
    def test_an_os_error_from_open_is_swallowed(self, recorder, method, args, kwargs):
        """Simulated ENOSPC / EACCES on every write path."""
        with patch("builtins.open", side_effect=OSError(28, "No space left on device")):
            assert getattr(recorder, method)(*args, **kwargs) is None

    def test_an_os_error_from_copy_is_swallowed(self, recorder, tmp_path):
        source = tmp_path / "out.srt"
        source.write_text("x", encoding="utf-8")
        with patch("shutil.copy2", side_effect=OSError(28, "No space left on device")):
            assert recorder.copy_output(str(source)) is None

    def test_an_unserialisable_payload_is_swallowed(self, recorder):
        """``default=str`` handles most objects; a self-referencing one still must not raise."""
        cycle = {}
        cycle["self"] = cycle
        assert recorder.save_cues("pre_translation", [cycle]) is None

    def test_a_broken_ffprobe_is_swallowed(self, recorder, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 32)
        with patch("subprocess.run", side_effect=OSError("ffprobe: not found")):
            assert recorder.record_source_video(str(video)) is None
        # ...and the run still finishes and indexes.
        recorder.finish(success=True)
        assert os.path.exists(os.path.join(recorder.run_dir, "meta.json"))

    def test_a_failing_index_append_does_not_lose_meta_json(self, recorder):
        real_open = open

        def selective(path, *args, **kwargs):
            if str(path).endswith("index.jsonl"):
                raise OSError(13, "Permission denied")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective):
            recorder.finish(success=True)
        assert os.path.exists(os.path.join(recorder.run_dir, "meta.json"))

    def test_start_run_never_raises_even_with_a_broken_config(self):
        with patch.object(rr, "ResearchRecorder", side_effect=RuntimeError("boom")):
            rec = rr.start_run("x", root="/tmp", enabled=True)
        assert rec.active is False
        assert rec.finish(success=True) is None

    def test_a_none_task_id_is_tolerated(self, root):
        rec = rr.start_run(None, root=root, enabled=True)
        rec.finish(success=True)
        assert rec.task_id == "unknown"


# ======================================================================================
@pytest.mark.unit
class TestKillSwitch:
    def test_disabled_writes_nothing_at_all(self, root):
        rec = rr.start_run("abcdef1234", root=root, enabled=False)
        assert rec.active is False
        rec.update_meta(pipeline="v2")
        rec.save_segments([{"text": "x"}])
        rec.finish(success=True)
        assert os.listdir(root) == []

    def test_config_supplies_the_defaults(self, root):
        class FakeConfig:
            RESEARCH_DIR = root
            RESEARCH_RECORDER_ENABLED = True
            RESEARCH_MAX_COPY_MB = 7

        with patch("config.get_config", return_value=FakeConfig()):
            rec = rr.start_run("cfg00000")
        assert rec.active is True
        assert rec.run_dir.startswith(root)
        assert rec.max_copy_mb == 7

    def test_config_kill_switch_is_honoured(self, root):
        class FakeConfig:
            RESEARCH_DIR = root
            RESEARCH_RECORDER_ENABLED = False
            RESEARCH_MAX_COPY_MB = 4096

        with patch("config.get_config", return_value=FakeConfig()):
            rec = rr.start_run("cfg00000")
        assert rec.active is False
        assert os.listdir(root) == []

    def test_the_default_is_on(self):
        """Owner's call: recording is the default, not an opt-in."""
        from config import Config

        assert Config.RESEARCH_RECORDER_ENABLED is True
        assert Config.RESEARCH_DIR.endswith("/research")


# ======================================================================================
@pytest.mark.unit
class TestSourceVideoProbe:
    def test_geometry_duration_and_fps_are_recorded(self, recorder, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 4096)

        class Result:
            returncode = 0
            stdout = json.dumps(
                {
                    "streams": [
                        {"width": 720, "height": 1280, "avg_frame_rate": "30000/1001"}
                    ],
                    "format": {"duration": "61.5"},
                }
            )

        with patch("subprocess.run", return_value=Result()):
            recorder.record_source_video(str(video))
        recorder.finish(success=True)

        meta = read_json(os.path.join(recorder.run_dir, "meta.json"))
        assert (meta["video_width"], meta["video_height"]) == (720, 1280)
        assert meta["video_fps"] == 29.97
        assert meta["video_duration_s"] == 61.5
        assert meta["source_name"] == "clip.mp4"
        assert meta["source_size_mb"] == 0.0

    def test_a_nonzero_ffprobe_is_noted_not_raised(self, recorder, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00")

        class Result:
            returncode = 1
            stdout = ""

        with patch("subprocess.run", return_value=Result()):
            recorder.record_source_video(str(video))
        recorder.finish(success=True)
        warnings = read_json(os.path.join(recorder.run_dir, "meta.json"))[
            "recorder_warnings"
        ]
        assert any("ffprobe" in w for w in warnings)


# ======================================================================================
# P4 — dropped-cue archiving
# ======================================================================================
@pytest.mark.unit
class TestSaveDroppedCues:
    """A dropped cue leaves no trace in the SRTs, the .ass or the video. If the archive
    does not record it, the hallucination gate's false-positive rate is unauditable."""

    def test_dropped_cues_are_written_with_their_reason(self, tmp_path):
        recorder = rr.start_run("task-dropped", root=str(tmp_path), enabled=True)
        recorder.save_dropped_cues(
            [{"start": 1.0, "end": 1.2, "text": "invented", "cps": 219.0}],
            reason="source_cps_hallucination",
        )
        recorder.finish(success=True)

        payload = json.loads(
            open(
                os.path.join(recorder.run_dir, "dropped_cues.json"), encoding="utf-8"
            ).read()
        )
        assert payload["reason"] == "source_cps_hallucination"
        assert payload["cues"][0]["text"] == "invented"
        assert payload["cues"][0]["cps"] == 219.0

    def test_count_and_reason_land_in_meta(self, tmp_path):
        recorder = rr.start_run("task-meta", root=str(tmp_path), enabled=True)
        recorder.save_dropped_cues([{"start": 0, "end": 1, "text": "x"}])
        recorder.finish(success=True)

        meta = json.loads(
            open(os.path.join(recorder.run_dir, "meta.json"), encoding="utf-8").read()
        )
        assert meta["dropped_cues_count"] == 1
        assert meta["dropped_cues_reason"] == "hallucination"

    def test_nothing_dropped_writes_no_file(self, tmp_path):
        """An absent file must mean "nothing was dropped", never "possibly unrecorded"."""
        recorder = rr.start_run("task-none", root=str(tmp_path), enabled=True)
        recorder.save_dropped_cues([])
        recorder.save_dropped_cues(None)
        recorder.finish(success=True)

        assert not os.path.exists(os.path.join(recorder.run_dir, "dropped_cues.json"))
        meta = json.loads(
            open(os.path.join(recorder.run_dir, "meta.json"), encoding="utf-8").read()
        )
        assert "dropped_cues_count" not in meta

    def test_input_is_not_mutated(self, tmp_path):
        recorder = rr.start_run("task-nomutate", root=str(tmp_path), enabled=True)
        cues = [{"start": 0, "end": 1, "text": "x"}]
        recorder.save_dropped_cues(cues)
        assert cues == [{"start": 0, "end": 1, "text": "x"}]

    def test_a_broken_payload_cannot_fail_the_job(self, tmp_path):
        """The recorder's whole contract: observability never takes production down."""
        recorder = rr.start_run("task-broken", root=str(tmp_path), enabled=True)
        assert recorder.save_dropped_cues(["not a dict"]) is None
        recorder.finish(success=True)
        assert os.path.exists(os.path.join(recorder.run_dir, "meta.json"))

    def test_disabled_recorder_is_a_noop(self, tmp_path):
        recorder = rr.start_run("task-off", root=str(tmp_path), enabled=False)
        assert recorder.save_dropped_cues([{"start": 0, "end": 1, "text": "x"}]) is None
