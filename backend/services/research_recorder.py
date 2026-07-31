"""
research_recorder — a complete, immutable archive of every processing run.

Why this exists
---------------
The pipeline writes its outputs to ``/app/downloads`` under names derived from the
SOURCE FILENAME: ``IMG_8975_translated.srt``, ``IMG_8975_final.mp4``. Run the same
video twice — different flags, different model, different prompt — and the second run
silently destroys the first. There is therefore no way to answer "did translation_v2
actually beat the legacy path on this clip", which is the question the whole
subtitle-quality branch exists to answer.

So: from now on every run is archived whole, under a name that cannot collide, on the
volume that survives rebuilds:

    <RESEARCH_DIR>/<UTC yyyymmdd_HHMMSS>_<task_id[:8]>/
        meta.json                  flags, model, dimensions, timings, tokens, cost, result
        segments.json              raw Whisper segments
        words.json                 word timestamps (v2 spotting runs only)
        cues_pre_translation.json  cue list handed to the translator
        cues_post_translation.json cue list it handed back (after the CPS pass)
        cues_post_reflow.json      final cue list, after the dangling-connector reflow
        llm/NN_system.txt          every OpenAI request, in call order
        llm/NN_user.txt
        llm/NN_response.json       the raw response object, not our parse of it
        llm/NN_meta.json           model, tokens, latency, error
        outputs/                   copies of the SRTs, the .ass and the final MP4
    <RESEARCH_DIR>/index.jsonl     one line per run, for cheap scanning

The safety contract (this is the important part)
------------------------------------------------
**The recorder may never fail a job and may never meaningfully slow one.** It is
observability, and observability that can take production down is worse than no
observability. Concretely:

* every public method is individually wrapped — one failure does not disable the rest;
* every failure is logged at WARNING and swallowed. Nothing propagates, ever;
* if the run directory cannot be created, :func:`start_run` returns a recorder whose
  methods are all no-ops, and the job proceeds unrecorded;
* ``config.RESEARCH_RECORDER_ENABLED`` is a kill switch that returns the same inert
  recorder without touching the filesystem at all.

Callers therefore never branch on ``if recorder:`` and never wrap calls in ``try``.
:func:`start_run` always returns an object with the full API.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Subdirectory names inside a run directory.
LLM_DIR = "llm"
OUTPUTS_DIR = "outputs"

#: The scan index, one JSON object per line, at the root of RESEARCH_DIR.
INDEX_FILENAME = "index.jsonl"

_MB = 1024 * 1024


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class _NullRecorder:
    """The inert recorder: same API, does nothing, cannot fail.

    Returned when recording is disabled or when the run directory could not be
    created. It is a real object rather than ``None`` so that call sites stay
    branch-free — ``recorder.save_cues(...)`` is always legal.
    """

    active = False
    run_dir = None
    task_id = None

    def __getattr__(self, name: str):
        def _noop(*_args, **_kwargs):
            return None

        return _noop

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<research_recorder disabled>"


def _guard(method):
    """Wrap a recorder method so it can never raise and never runs while inactive."""

    def wrapper(self, *args, **kwargs):
        if not self.active:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the entire point of this decorator
            logger.warning(
                "research_recorder: %s failed (%s: %s) — run continues unrecorded "
                "for this item",
                method.__name__,
                type(exc).__name__,
                exc,
            )
            return None

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


class ResearchRecorder:
    """Archives one processing run. Every method is failure-proof — see module docs."""

    def __init__(self, task_id: str, root: str, max_copy_mb: int = 4096):
        self.task_id = str(task_id or "unknown")
        self.root = root
        self.max_copy_mb = int(max_copy_mb)
        self.run_dir: str | None = None
        self.active = False
        self._llm_index = 0
        self._started_monotonic = time.time()
        self._meta: dict[str, Any] = {
            "task_id": self.task_id,
            "started_at": _iso_now(),
            "timings": {},
            "recorder_warnings": [],
        }

        # Directory creation is the one step that must happen up front; if it fails the
        # recorder simply stays inactive and the job never notices.
        try:
            name = f"{_utc_stamp()}_{self.task_id[:8]}"
            run_dir = os.path.join(self.root, name)
            os.makedirs(os.path.join(run_dir, LLM_DIR), exist_ok=True)
            os.makedirs(os.path.join(run_dir, OUTPUTS_DIR), exist_ok=True)
            self.run_dir = run_dir
            self._meta["dir"] = name
            self.active = True
            logger.info("research_recorder: archiving run to %s", run_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "research_recorder: could not create run directory under %s (%s: %s) "
                "— this run will NOT be archived; processing continues",
                self.root,
                type(exc).__name__,
                exc,
            )

    # -- internals ------------------------------------------------------------------

    def _path(self, *parts: str) -> str:
        return os.path.join(self.run_dir, *parts)

    def _write_json(self, relative: str, payload: Any) -> None:
        with open(self._path(relative), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    def _note(self, message: str) -> None:
        """Record a recorder-level problem inside meta.json (never raises)."""
        try:
            self._meta.setdefault("recorder_warnings", []).append(message)
        except Exception:  # pragma: no cover - defensive
            pass

    # -- metadata -------------------------------------------------------------------

    @_guard
    def update_meta(self, **fields: Any) -> None:
        """Merge fields into meta.json's in-memory buffer (flushed by :meth:`finish`)."""
        self._meta.update(fields)

    @_guard
    def add_timing(self, stage: str, seconds: float) -> None:
        """Record how long one pipeline stage took, in seconds."""
        self._meta.setdefault("timings", {})[str(stage)] = round(float(seconds), 3)

    @_guard
    def record_source_video(self, video_path: str) -> None:
        """ffprobe the source once and store its geometry, duration and frame rate.

        Deliberately the recorder's own probe rather than a value threaded down from
        the pipeline: the archive has to describe the run even when the pipeline never
        needed the number, and one 50ms ffprobe per job is not a cost worth coupling for.
        """
        self._meta["source_path"] = video_path
        self._meta["source_name"] = os.path.basename(video_path or "")
        try:
            self._meta["source_size_mb"] = round(os.path.getsize(video_path) / _MB, 2)
        except OSError:
            self._meta["source_size_mb"] = None

        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", "-select_streams", "v:0", video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            self._note(f"ffprobe failed with code {result.returncode}")
            return
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if streams:
            stream = streams[0]
            self._meta["video_width"] = stream.get("width")
            self._meta["video_height"] = stream.get("height")
            rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
            if "/" in rate:
                num, _, den = rate.partition("/")
                try:
                    self._meta["video_fps"] = (
                        round(float(num) / float(den), 3) if float(den) else None
                    )
                except (ValueError, ZeroDivisionError):
                    self._meta["video_fps"] = None
        duration = (data.get("format") or {}).get("duration")
        try:
            self._meta["video_duration_s"] = round(float(duration), 3) if duration else None
        except (TypeError, ValueError):
            self._meta["video_duration_s"] = None

    # -- pipeline artefacts ---------------------------------------------------------

    @_guard
    def save_segments(self, segments: Any) -> None:
        """Archive the raw transcription segments."""
        self._write_json("segments.json", list(segments or []))
        self._meta["segments_count"] = len(list(segments or []))

    @_guard
    def save_words(self, words: Any) -> None:
        """Archive word-level timestamps. No-op when the run produced none."""
        words = list(words or [])
        if not words:
            return
        self._write_json("words.json", words)
        self._meta["words_count"] = len(words)

    @_guard
    def save_cues(self, stage: str, cues: Any) -> None:
        """Archive a cue list at one named pipeline stage.

        ``stage`` becomes the filename: ``pre_translation`` -> ``cues_pre_translation.json``.
        """
        cues = [dict(cue) for cue in (cues or [])]
        self._write_json(f"cues_{stage}.json", cues)
        self._meta.setdefault("cue_counts", {})[str(stage)] = len(cues)

    @_guard
    def record_llm(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        response: Any = None,
        meta: dict | None = None,
    ) -> None:
        """Archive one OpenAI request/response pair, numbered in call order.

        Writes ``NN_system.txt``, ``NN_user.txt``, ``NN_response.json`` and
        ``NN_meta.json`` into ``llm/``. The response is stored RAW (the SDK object's own
        ``model_dump``), not the parsed cue map — the parse is one of the things a corpus
        analysis needs to be able to second-guess.

        A failed request is recorded too, with ``error`` in its meta and no response file.
        """
        self._llm_index += 1
        prefix = f"{self._llm_index:02d}"
        with open(self._path(LLM_DIR, f"{prefix}_system.txt"), "w", encoding="utf-8") as fh:
            fh.write(system or "")
        with open(self._path(LLM_DIR, f"{prefix}_user.txt"), "w", encoding="utf-8") as fh:
            fh.write(user or "")

        if response is not None:
            payload: Any
            try:
                payload = response.model_dump()  # openai>=1.x pydantic models
            except Exception:  # noqa: BLE001 - test doubles and older SDKs
                try:
                    payload = response.to_dict()
                except Exception:  # noqa: BLE001
                    payload = {"repr": repr(response)}
            self._write_json(os.path.join(LLM_DIR, f"{prefix}_response.json"), payload)

        record = {"stage": str(stage), "index": self._llm_index, "at": _iso_now()}
        record.update(meta or {})
        self._write_json(os.path.join(LLM_DIR, f"{prefix}_meta.json"), record)

    # -- outputs --------------------------------------------------------------------

    @_guard
    def copy_output(self, path: str, alias: str | None = None) -> None:
        """Copy one produced file into ``outputs/``.

        Files above ``max_copy_mb`` are skipped with a WARNING rather than filling the
        volume; the skip is recorded in meta.json so the archive never lies about what
        it holds.
        """
        if not path or not os.path.exists(path):
            return
        size_mb = os.path.getsize(path) / _MB
        name = alias or os.path.basename(path)
        if size_mb > self.max_copy_mb:
            logger.warning(
                "research_recorder: %s is %.1f MB (> %d MB) — not archived",
                name,
                size_mb,
                self.max_copy_mb,
            )
            self._note(f"{name} skipped: {size_mb:.1f} MB over the {self.max_copy_mb} MB limit")
            return
        shutil.copy2(path, self._path(OUTPUTS_DIR, name))
        self._meta.setdefault("outputs", {})[name] = round(size_mb, 2)

    @_guard
    def copy_outputs(self, paths: dict) -> None:
        """Copy several outputs at once: ``{alias_or_None: path}`` or ``{path: None}``."""
        for alias, path in (paths or {}).items():
            if path is None:
                self.copy_output(alias)
            else:
                self.copy_output(path, alias if isinstance(alias, str) else None)

    # -- completion -----------------------------------------------------------------

    @_guard
    def finish(self, success: bool, error: str | None = None) -> None:
        """Flush meta.json and append the run's line to ``index.jsonl``.

        Called from the task's ``finally``, so it runs for failed jobs too — a failure
        with its prompts and partial cues attached is one of the more valuable rows in
        the corpus.
        """
        self._meta["success"] = bool(success)
        self._meta["error"] = error
        self._meta["finished_at"] = _iso_now()
        self._meta["wall_seconds"] = round(time.time() - self._started_monotonic, 3)
        self._meta["llm_requests"] = self._llm_index
        self._write_json("meta.json", self._meta)
        self._append_index()

    def _append_index(self) -> None:
        """Append one compact line to RESEARCH_DIR/index.jsonl.

        A single ``write`` of one short line to an ``O_APPEND`` handle is atomic on
        POSIX below PIPE_BUF, which is why there is no lock here: concurrent workers
        interleave whole lines, never fragments.
        """
        try:
            row = {
                "task_id": self.task_id,
                "dir": self._meta.get("dir"),
                "at": self._meta.get("finished_at"),
                "pipeline": self._meta.get("pipeline"),
                "spotting_v2": self._meta.get("spotting_v2"),
                "translation_v2": self._meta.get("translation_v2"),
                "translation_style": self._meta.get("translation_style"),
                "render_v2": self._meta.get("render_v2"),
                "whisper_model": self._meta.get("whisper_model"),
                "translation_service": self._meta.get("translation_service"),
                "video": self._meta.get("source_name"),
                "w": self._meta.get("video_width"),
                "h": self._meta.get("video_height"),
                "duration_s": self._meta.get("video_duration_s"),
                "cues": (self._meta.get("cue_counts") or {}).get("post_reflow")
                or self._meta.get("segments_count"),
                "success": self._meta.get("success"),
                "tokens": self._meta.get("translation_tokens"),
                "cost_usd": self._meta.get("translation_cost_usd"),
            }
            line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
            with open(os.path.join(self.root, INDEX_FILENAME), "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "research_recorder: could not append to %s (%s: %s)",
                INDEX_FILENAME,
                type(exc).__name__,
                exc,
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ResearchRecorder task={self.task_id[:8]} dir={self.run_dir}>"


def start_run(task_id: str, *, root: str = None, enabled: bool = None) -> Any:
    """Begin archiving a run. **Always returns a usable recorder — never raises.**

    Args:
        task_id: the Celery task id; its first 8 characters name the run directory.
        root: archive root. Defaults to ``config.RESEARCH_DIR``.
        enabled: override the ``RESEARCH_RECORDER_ENABLED`` kill switch.

    Returns:
        A :class:`ResearchRecorder`, or an inert stand-in with the identical API when
        recording is switched off or the directory could not be made. Callers must not
        test the result for truthiness — both are truthy and both are safe to call.
    """
    try:
        if root is None or enabled is None:
            from config import get_config

            config = get_config()
            if root is None:
                root = getattr(config, "RESEARCH_DIR", "/app/storage/research")
            if enabled is None:
                enabled = getattr(config, "RESEARCH_RECORDER_ENABLED", True)
            max_copy_mb = getattr(config, "RESEARCH_MAX_COPY_MB", 4096)
        else:
            max_copy_mb = 4096

        if not enabled:
            logger.info("research_recorder: disabled (RESEARCH_RECORDER_ENABLED=false)")
            return _NullRecorder()

        recorder = ResearchRecorder(task_id, root, max_copy_mb=max_copy_mb)
        return recorder if recorder.active else _NullRecorder()
    except Exception as exc:  # noqa: BLE001 - start_run itself must not be able to fail
        logger.warning(
            "research_recorder: start_run failed (%s: %s) — running unrecorded",
            type(exc).__name__,
            exc,
        )
        return _NullRecorder()
